#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import copy
from functools import partial
from itertools import product
from multiprocessing.pool import ThreadPool
from threading import Lock
import Queue
import time
import numpy as np

from pyspark.ml import Pipeline
from pyspark.ml.base import Estimator
from pyspark.ml.param.shared import HasParallelism
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.sql.functions import rand

__all__ = ['DagCrossValidator', 'DagPipeline']


class PipelineTreeNode(object):

    def __init__(self, stage, param_map, tree_type):
        self.parent = None
        self.children = []
        self.stage = stage
        self.param_map = param_map
        self.transformer = None
        self.lock = None
        self.cached_count = 0
        self.tree_type = tree_type

    def __repr__(self):
        param_map_str = str({p.name: v for p, v in self.param_map.iteritems()})
        parent_map_str = str([{} if self.parent is None else
                              {p.name: v for p, v in self.parent.param_map.iteritems()}])
        return self.stage.uid + param_map_str + parent_map_str

    def cache_task(self, task, dataset):
        if not dataset.is_cached:
            dataset.cache()
        task(dataset)
        with self.lock:
            self.cached_count -= 1
            if self.cached_count == 0:
                dataset.unpersist()

    def init_is_caching(self):
        is_caching = len(self.children) > 0 and \
                     isinstance(self.children[0], self.tree_type.EstimatorNode) and \
                     not isinstance(self.children[0], self.tree_type.FeatureExtractionEstimatorNode)
        if is_caching:
            if self.lock is None:
                self.lock = Lock()
            with self.lock:
                self.cached_count = len(self.children)
        return is_caching

    def get_cache_task(self, task):
        return partial(self.cache_task, task)


class BFSTree(object):

    class TransformerNode(PipelineTreeNode):

        def __init__(self, stage, param_map):
            super(BFSTree.TransformerNode, self).__init__(stage, param_map, BFSTree)

        def transform_task(self, queue, results, dataset):
            transformed_dataset = self.transformer.transform(dataset)
            is_caching = self.init_is_caching()
            for child in self.children:
                task = child.get_fit_task(queue, results)
                if is_caching:
                    task = self.get_cache_task(task)
                queue.put(partial(task, transformed_dataset))

        def evaluate_task(self, queue, eval, results, dataset):
            transformed_dataset = self.transformer.transform(dataset)
            if self.children:
                is_caching = self.init_is_caching()
                for child in self.children:
                    task = child.get_evaluate_task(queue, eval, results)
                    if is_caching:
                        task = self.get_cache_task(task)
                    queue.put(partial(task, transformed_dataset))
            else:
                metric = eval.evaluate(transformed_dataset)
                results.put((metric, self))
                if results.full():
                    queue.stop()

        def get_fit_task(self, queue, results):
            self.transformer = self.stage.copy(self.param_map)
            return partial(self.transform_task, queue, results)

        def get_evaluate_task(self, queue, eval, results):
            return partial(self.evaluate_task, queue, eval, results)

    class EstimatorNode(TransformerNode):

        def fit_task(self, queue, results, dataset):
            self.transformer = self.stage.fit(dataset, params=self.param_map)
            # The last estimator could only have children that transform, not fit
            results.put(self)
            if results.full():
                queue.stop()

        def get_fit_task(self, queue, results):
            return partial(self.fit_task, queue, results)

    class FeatureExtractionEstimatorNode(EstimatorNode):

        def fit_task(self, queue, results, dataset):
            self.transformer = self.stage.fit(dataset, params=self.param_map)
            transformed_dataset = self.transformer.transform(dataset)
            if self.children:
                is_caching = self.init_is_caching()
                for child in self.children:
                    task = child.get_fit_task(queue, results)
                    if is_caching:
                        task = self.get_cache_task(task)
                    queue.put(partial(task, transformed_dataset))
            else:
                results.put(self)
                if results.full():
                    queue.stop()

        def get_fit_task(self, queue, results):
            return partial(self.fit_task, queue, results)


class DFSTree(object):

    class TransformerNode(PipelineTreeNode):

        def __init__(self, stage, param_map):
            super(DFSTree.TransformerNode, self).__init__(stage, param_map, DFSTree)

        def transform_task(self, queue, results, holding, dataset):
            transformed_dataset = self.transformer.transform(dataset)
            if self.children:
                is_caching = self.init_is_caching()
                for i, child in enumerate(self.children):
                    task = child.get_fit_task(queue, results, holding)
                    if is_caching:
                        task = self.get_cache_task(task)
                    if i == 0 or not child.children:
                        queue.put(partial(task, transformed_dataset))
                    else:
                        holding.put(partial(task, transformed_dataset))
            elif not holding.empty():
                queue.put(holding.get())

        def evaluate_task(self, queue, eval, results, holding, dataset):
            transformed_dataset = self.transformer.transform(dataset)
            if self.children:
                is_caching = self.init_is_caching()
                for i, child in enumerate(self.children):
                    task = child.get_evaluate_task(queue, eval, results, holding)
                    if is_caching:
                        task = self.get_cache_task(task)
                    if i == 0 or not child.children:
                        queue.put(partial(task, transformed_dataset))
                    else:
                        holding.put(partial(task, transformed_dataset))
            else:
                metric = eval.evaluate(transformed_dataset)
                results.put((metric, self))
                if not holding.empty():
                    queue.put(holding.get())
                if results.full():
                    queue.stop()

        def get_fit_task(self, queue, results, holding=Queue.Queue()):
            self.transformer = self.stage.copy(self.param_map)
            return partial(self.transform_task, queue, results, holding)

        def get_evaluate_task(self, queue, eval, results, holding=Queue.Queue()):
            return partial(self.evaluate_task, queue, eval, results, holding)

    class EstimatorNode(TransformerNode):

        def fit_task(self, queue, results, holding, dataset):
            self.transformer = self.stage.fit(dataset, params=self.param_map)
            # The last estimator could only have children that transform, not fit
            if not holding.empty():
                queue.put(holding.get())
            results.put(self)
            if results.full():
                queue.stop()

        def get_fit_task(self, queue, results, holding=Queue.Queue()):
            return partial(self.fit_task, queue, results, holding)

    class FeatureExtractionEstimatorNode(EstimatorNode):

        def fit_task(self, queue, results, holding, dataset):
            self.transformer = self.stage.fit(dataset, params=self.param_map)
            transformed_dataset = self.transformer.transform(dataset)
            if self.children:
                is_caching = self.init_is_caching()
                for i, child in enumerate(self.children):
                    task = child.get_fit_task(queue, results, holding)
                    if is_caching:
                        task = self.get_cache_task(task)
                    if i == 0 or not child.children:
                        queue.put(partial(task, transformed_dataset))
                    else:
                        holding.put(partial(task, transformed_dataset))
            else:
                results.put(self)
                if results.full():
                    queue.stop()

        def get_fit_task(self, queue, results, holding=Queue.Queue()):
            return partial(self.fit_task, queue, results, holding)


class IterableQueue(Queue.Queue):

    _sentinel = object()

    def __init__(self, maxsize=0):
        Queue.Queue.__init__(self, maxsize)

    def __iter__(self):
        return iter(self.get, self._sentinel)

    def stop(self):
        self.put(self._sentinel)


class DagPipeline(Pipeline, HasParallelism):

    def __init__(self, stages, parallelism, tree_type="bfs"):
        super(DagPipeline, self).__init__(stages=stages)
        self.setParallelism(parallelism)
        self.roots = None
        self.nodes = None
        self.tree_type = tree_type

    def evaluate(self, paramMaps, train, validation, eval):
        num_models = len(paramMaps)
        print('Num Models: %d, Parallelism: %d' % (num_models, self.getParallelism()))

        # Build a DAG from the stages
        self.roots, self.nodes = self.build_dag(paramMaps)

        # Run tasks in queue using thread pool
        pool = ThreadPool(processes=min(self.getParallelism(), num_models))
        queue = IterableQueue()

        # Queue to store results
        results = Queue.Queue(maxsize=num_models)

        start = time.time()

        # Fit the pipeline models
        for root in self.roots:
            task = root.get_fit_task(queue, results)
            queue.put(partial(task, train))

        for _ in pool.imap_unordered(lambda f: f(), queue):
            pass
        #for task in queue:
        #    task()

        elapsed = time.time() - start
        print("Time to fit: %s" % elapsed)

        # clear results queue
        results = Queue.Queue(maxsize=num_models)

        start = time.time()

        # Evaluate the modesl
        for root in self.roots:
            task = root.get_evaluate_task(queue, eval, results)
            queue.put(partial(task, validation))

        for _ in pool.imap_unordered(lambda f: f(), queue):
            pass
        #for task in queue:
        #    task()

        elapsed = time.time() - start
        print("Time to eval: %s" % elapsed)

        # Gather metrics and put back in order of original epm list
        metrics_params = []
        while not results.empty():
            metric, node = results.get()
            model_params = {}
            while node is not None:
                model_params.update(node.param_map)
                node = node.parent
            metrics_params.append((metric, model_params))
        orig_order = {str(e): i for e, i in  zip(paramMaps, range(num_models))}
        metrics_params = sorted(metrics_params, key=lambda mp: orig_order[str(mp[1])])
        metrics = [metric for metric, model_params in metrics_params]

        return metrics

    def build_dag(self, paramMaps):

        # Type of tree search to execute dag
        tree = BFSTree if self.tree_type.lower() == "bfs" else DFSTree

        stages = self.getStages()
        stage_prev = None

        # Locate the last estimator, will not transform training data after fit
        for i, stage in enumerate(stages):
            if isinstance(stage, Estimator):
                last_est_index = i

        nodes = []
        roots = []
        stage_nodes = {}
        for i, stage in enumerate(stages):

            # Determine what type of Node for the stage
            if isinstance(stage, Estimator):
                if i < last_est_index:
                    Node = tree.FeatureExtractionEstimatorNode
                else:
                    Node = tree.EstimatorNode
            else:
                Node = tree.TransformerNode

            # Separate ParamMaps for the stage
            temp_map = {}
            for param_map in paramMaps:
                for k, v in param_map.iteritems():
                    if k.parent == stage.uid:
                        temp_map.setdefault(k, set()).add(v)

            # Check if have a param grid for this stage
            if temp_map:
                grid_builder = ParamGridBuilder()
                for k, v in temp_map.iteritems():
                    grid_builder.addGrid(k, v)
                stage_param_grid = grid_builder.build()
                new_nodes = [Node(stage, param_map) for param_map in stage_param_grid]
            else:
                new_nodes = [Node(stage, {})]

            # Make nodes for each node of parent stage
            if stage_prev:
                temp_nodes = []
                parent_nodes = stage_nodes[stage_prev]
                for parent_node in parent_nodes:
                    for node in new_nodes:
                        child_node = copy.copy(node)
                        child_node.children = []
                        child_node.parent = parent_node
                        parent_node.children.append(child_node)
                        temp_nodes.append(child_node)
                new_nodes = temp_nodes
            else:
                roots += new_nodes

            # Store all new nodes created for this stage
            stage_nodes[stage] = new_nodes
            nodes += new_nodes

            stage_prev = stage

        return roots, nodes

    def get_graph(self, nodes=None, draw=False):
        try:
            import networkx as nx

            if nodes is None:
                if self.nodes is None:
                    raise RuntimeError("Must pass in Nodes or evaluate pipeline first")
                else:
                    nodes = self.nodes

            g = nx.DiGraph()
            for node in nodes:
                if node.parent is not None:
                    g.add_edge(node.parent, node)
            if draw:
                nx.draw(g, with_labels=True)

            return g
        except ImportError as e:
            raise ImportError("Must install networkx to build graph\n%s" % e)


class DagCrossValidator(CrossValidator):

    def _fit(self, dataset):
        current_estimator = self.getEstimator()

        # Not a Pipeline, use standard CrossValidator
        if not isinstance(current_estimator, Pipeline):
            return super(DagCrossValidator, self)._fit(dataset)
        # Delegate parallelism to DagPipeline
        elif not isinstance(current_estimator, DagPipeline):
            dag_pipeline = DagPipeline(stages=current_estimator.getStages(),
                                       parallelism=self.getParallelism())
        # Already a DagPipeline
        else:
            dag_pipeline = current_estimator

        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
        eva = self.getOrDefault(self.evaluator)
        nFolds = self.getOrDefault(self.numFolds)
        seed = self.getOrDefault(self.seed)
        h = 1.0 / nFolds
        randCol = self.uid + "_rand"
        df = dataset.select("*", rand(seed).alias(randCol))
        metrics = [0.0] * numModels

        for i in range(nFolds):
            validateLB = i * h
            validateUB = (i + 1) * h
            condition = (df[randCol] >= validateLB) & (df[randCol] < validateUB)
            validation = df.filter(condition).cache()
            train = df.filter(~condition).cache()

            fold_metrics = dag_pipeline.evaluate(epm, train, validation, eva)

            for j in range(len(metrics)):
                metrics[j] += fold_metrics[j] / nFolds

            validation.unpersist()
            train.unpersist()

        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)

        bestModel = current_estimator.fit(dataset, epm[bestIndex])

        return self._copyValues(CrossValidatorModel(bestModel, metrics))
