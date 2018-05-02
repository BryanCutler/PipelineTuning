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
import Queue
import time
import numpy as np

from pyspark.ml import Pipeline
from pyspark.ml.base import Estimator
from pyspark.ml.param.shared import HasParallelism
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.sql.functions import rand

__all__ = ['DagCrossValidator']


class BFSTree(object):

    class TransformerNode(object):

        def __init__(self, stage, stage_inputs, param_map):
            self.parents = []
            self.children = []
            self.stage = stage
            self.stage_inputs = stage_inputs
            self.param_map = param_map
            self.transformer = None

        def __repr__(self):
            param_map_str = str({p.name: v for p, v in self.param_map.iteritems()})
            parent_map_str = str([{p.name: v for p, v in parent.param_map.iteritems()}
                                  for parent in self.parents])
            return self.stage.uid + param_map_str + parent_map_str

        def copy_with_parent_params(self, param_map):
            node = copy.copy(self)
            node.parent_param_map.update(param_map)
            return node

        def transform_task(self, queue, results, dataset):
            transformed_dataset = self.transformer.transform(dataset)
            for child in self.children:
                task = child.get_fit_task(queue, results)
                queue.put(partial(task, transformed_dataset))

        def evaluate_task(self, queue, eval, results, dataset):
            transformed_dataset = self.transformer.transform(dataset)
            if self.children:
                for child in self.children:
                    task = child.get_evaluate_task(queue, eval, results)
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
            if not dataset.is_cached:
                dataset.cache()
            self.transformer = self.stage.fit(dataset, params=self.param_map)
            if self.children:
                raise NotImplementedError("Estimator children")
            else:
                results.put(self)
                if results.full():
                    queue.stop()

        def evaluate_task(self, queue, eval, results, dataset):
            if not dataset.is_cached:
                dataset.cache()
            super(BFSTree.EstimatorNode, self).evaluate_task(queue, eval, results, dataset)

        def get_fit_task(self, queue, results):
            return partial(self.fit_task, queue, results)

    class FeatureExtractionEstimatorNode(EstimatorNode):

        def fit_task(self, queue, results, dataset):
            self.transformer = self.stage.fit(dataset, params=self.param_map)
            transformed_dataset = self.transformer.transform(dataset)
            if self.children:
                for child in self.children:
                    task = child.get_fit_task(queue, results)
                    queue.put(partial(task, transformed_dataset))
            else:
                raise NotImplementedError("feature ext no children")

        def get_fit_task(self, queue, results):
            return partial(self.fit_task, queue, results)


class DFSTree(object):

    class TransformerNode(object):

        def __init__(self, stage, stage_inputs, param_map):
            self.parents = []
            self.children = []
            self.stage = stage
            self.stage_inputs = stage_inputs
            self.param_map = param_map
            self.transformer = None

        def __repr__(self):
            param_map_str = str({p.name: v for p, v in self.param_map.iteritems()})
            parent_map_str = str([{p.name: v for p, v in parent.param_map.iteritems()}
                                  for parent in self.parents])
            return self.stage.uid + param_map_str + parent_map_str

        def copy_with_parent_params(self, param_map):
            node = copy.copy(self)
            node.parent_param_map.update(param_map)
            return node

        def transform_task(self, queue, results, task_stack, dataset):
            transformed_dataset = self.transformer.transform(dataset)
            if self.children:
                for i, child in enumerate(self.children):
                    task = child.get_fit_task(queue, results, task_stack)
                    if i == 0 or not child.children:
                        queue.put(partial(task, transformed_dataset))
                    else:
                        task_stack.insert(0, partial(task, transformed_dataset))
            elif task_stack:
                queue.put(task_stack.pop(0))

        def evaluate_task(self, queue, eval, results, task_stack, dataset):
            transformed_dataset = self.transformer.transform(dataset)
            if self.children:
                for i, child in enumerate(self.children):
                    task = child.get_evaluate_task(queue, eval, results, task_stack)
                    if i == 0 or not child.children:
                        if not child.children and not transformed_dataset.is_cached:
                            transformed_dataset.cache()
                        queue.put(partial(task, transformed_dataset))
                    else:
                        task_stack.insert(0, partial(task, transformed_dataset))
            else:
                metric = eval.evaluate(transformed_dataset)
                results.put((metric, self))
                if task_stack:
                    queue.put(task_stack.pop(0))
                if results.full():
                    queue.stop()

        def get_fit_task(self, queue, results, task_stack=[]):
            self.transformer = self.stage.copy(self.param_map)
            return partial(self.transform_task, queue, results, task_stack)

        def get_evaluate_task(self, queue, eval, results, task_stack=[]):
            return partial(self.evaluate_task, queue, eval, results, task_stack)

    class EstimatorNode(TransformerNode):

        def fit_task(self, queue, results, task_stack, dataset):
            self.transformer = self.stage.fit(dataset, params=self.param_map)
            if self.children:
                raise NotImplementedError("Estimator children")
            else:
                if task_stack:
                    queue.put(task_stack.pop(0))
                results.put(self)
                if results.full():
                    queue.stop()

        def get_fit_task(self, queue, results, task_stack=[]):
            return partial(self.fit_task, queue, results, task_stack)

    class FeatureExtractionEstimatorNode(EstimatorNode):

        def fit_task(self, queue, results, task_stack, dataset):
            self.transformer = self.stage.fit(dataset, params=self.param_map)
            transformed_dataset = self.transformer.transform(dataset)
            if self.children:
                for i, child in enumerate(self.children):
                    task = child.get_fit_task(queue, results, task_stack)
                    if i == 0 or not child.children:
                        if not child.children and not transformed_dataset.is_cached:
                            transformed_dataset.cache()
                        queue.put(partial(task, transformed_dataset))
                    else:
                        task_stack.insert(0, partial(task, transformed_dataset))
            else:
                raise NotImplementedError("feature ext no children")

        def get_fit_task(self, queue, results, task_stack=[]):
            return partial(self.fit_task, queue, results, task_stack)


class IterableQueue(Queue.Queue):

    _sentinel = object()

    def __init__(self, maxsize=0):
        Queue.Queue.__init__(self, maxsize)

    def __iter__(self):
        return iter(self.get, self._sentinel)

    def stop(self):
        self.put(self._sentinel)


class DagPipeline(Pipeline, HasParallelism):

    def __init__(self, stages, parallelism, tree_type="dfs"):
        super(DagPipeline, self).__init__(stages=stages)
        self.setParallelism(parallelism)
        self.roots = None
        self.nodes = None
        self.tree_type = tree_type

    def evaluate(self, paramMaps, train, validation, eval):
        num_models = len(paramMaps)
        print('Num Models: %d, Parallelism: %d' % (num_models, self.getParallelism()))

        if self.roots is None:
            self.roots, self.nodes = self.build_dag(paramMaps)

        pool = ThreadPool(processes=min(self.getParallelism(), num_models))

        queue = IterableQueue()

        results = Queue.Queue(maxsize=num_models)

        start = time.time()

        # Fit the pipeline models
        for root in self.roots:
            task = root.get_fit_task(queue, results)
            queue.put(partial(task, train))

        for result in pool.imap_unordered(lambda f: f(), queue):
            pass
        #for task in queue:
        #    task()

        elapsed = time.time() - start
        print("Time to fit: %s" % elapsed)

        def get_models_from_leaf(leaf):
            transformers = []
            curr = leaf
            while curr is not None:
                transformers.insert(0, curr.stage)
                curr = curr.parents[0] if curr.parents else None  # TODO: multiple parents
            return transformers

        # clear results queue
        while not results.empty():
            results.get()

        start = time.time()

        # Evaluate the modesl
        for root in self.roots:
            task = root.get_evaluate_task(queue, eval, results)
            queue.put(partial(task, validation))

        for result in pool.imap_unordered(lambda f: f(), queue):
            pass
        #for task in queue:
        #    task()

        elapsed = time.time() - start
        print("Time to eval: %s" % elapsed)

        metrics = []
        models = {}
        while not results.empty():
            metric, leaf = results.get()
            metrics.append(metric)
            #models[str(leaf)] = get_models_from_leaf(leaf)
            #print("Model: %s, Metric: %s" % (leaf, metric))

        return metrics

    def build_dag(self, paramMaps):

        # Type of tree search to execute dag
        tree = BFSTree if self.tree_type.lower() == "bfs" else DFSTree

        # maps to lookup param values for inputs/outputs
        input_lookups = [
            ("inputCol", lambda params: params.getInputCol()),
            ("featuresCol", lambda params: params.getFeaturesCol()),
            ("labelCol", lambda params: params.getLabelCol())
        ]
        output_lookups = [
            ("outputCol", lambda params: params.getOutputCol()),
            ("predictionCol", lambda params: params.getPredictionCol())
        ]

        stages = self.getStages()

        # Locate the last estimator, will not transform training data after fit
        for i, stage in enumerate(stages):
            if isinstance(stage, Estimator):
                last_est_index = i

        nodes = []
        roots = []
        stage_nodes = {}
        output_to_stage = {}
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

            # Get the inputs/outputs for the stage
            stage_inputs = [lookup[1](stage) for lookup in input_lookups
                            if stage.hasParam(lookup[0])]
            for lookup in output_lookups:
                if stage.hasParam(lookup[0]):
                    output_to_stage[lookup[1](stage)] = stage

            # Check if have a param grid for this stage
            if temp_map:
                grid_builder = ParamGridBuilder()
                for k, v in temp_map.iteritems():
                    grid_builder.addGrid(k, v)
                stage_param_grid = grid_builder.build()
                new_nodes = [Node(stage, stage_inputs, param_map) for param_map in stage_param_grid]
            else:
                new_nodes = [Node(stage, stage_inputs, {})]

            # Find parent stage, assume stages in topological order
            parent_stages = [output_to_stage[i] for i in stage_inputs if i in output_to_stage]

            # Make nodes for each node of parent stages
            if parent_stages:
                parent_nodes_list = [stage_nodes[parent_stage] for parent_stage in parent_stages]
                parent_nodes_prod = product(*parent_nodes_list)
                temp_nodes = []
                for parent_nodes in parent_nodes_prod:
                    for node in new_nodes:
                        child_node = copy.copy(node)
                        child_node.parents = parent_nodes
                        for parent_node in parent_nodes:
                            parent_node.children.append(child_node)
                        temp_nodes.append(child_node)
                new_nodes = temp_nodes
            else:
                roots += new_nodes

            # Save the nodes that produce the output
            stage_nodes[stage] = new_nodes

            nodes += new_nodes

        return roots, nodes

    def get_graph(self, nodes=None, draw=False):
        try:
            import networkx as nx

            if nodes is None and self.nodes is None:
                raise RuntimeError("Must pass in Nodes or evaluate pipeline first")

            g = nx.DiGraph()
            for node in self.nodes:
                for parent in node.parents:
                    g.add_edge(parent, node)

            #topo = list(nx.topological_sort(g))

            if draw:
                nx.draw(g, with_labels=True)

        except ImportError as e:
            raise ImportError("Must install networkx to build graph\n%s" % e)


class DagCrossValidator(CrossValidator):

    def _fit(self, dataset):
        current_estimator = self.getEstimator()

        #bestModel = super(DagCrossValidator, self)._fit(dataset)
        #print(bestModel.avgMetrics)
        #return bestModel

        # Not a Pipeline, use standard CrossValidator
        if not isinstance(current_estimator, Pipeline):
            return super(DagCrossValidator, self)._fit(dataset)

        # Delegate parallelism to DagPipeline
        dag_pipeline = DagPipeline(stages=current_estimator.getStages(),
                                   parallelism=self.getParallelism())
        #self.setEstimator(dag_pipeline)
        #self.setParallelism(1)
        #model = super(DagCrossValidator, self)._fit(dataset)

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

        #self.setParallelism(dag_pipeline.getParallelism())
        #self.setEstimator(current_estimator)

        return self._copyValues(CrossValidatorModel(bestModel, metrics))
