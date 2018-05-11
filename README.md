# Optimized Spark Python Cross-Validation for Pipelines

## Description

This module is an experimental drop-in replacement for the current Spark Python CrossValidator. It optimizes pipeline model selection over a parameter grid by eliminating duplicated work done when fitting and evaluating pipeline stages, while still allowing for the existing model parallelism set by the `parallelism` parameter. This is done by examining each stage of the pipeline along with the given parameter grid and building a Directed Acyclic Graph (DAG) where each node represents a model and/or transformer of a stage with a particular set of parameters. A path in the DAG is equivalent to a `PipelineModel`. Additionaly, nodes in the DAG can cache a DataFrame to allow all child nodes to reuse the parent transforms.  This is currently done in the parent node of the last estimator in the pipeline, which is usually the predictor stage that has the largest parameter grid and can benefit the most by cached input.

## Usage

Trying out this module in your Spark cluster is simple because it only is needed for the driver program itself. Just download the `dag_tuning.py` file and place in the same directory as your application. In your application, just replace the `CrossValidator` class with `DagCrossValidator`.  Usage will then be automatic when tuning a pipeline.

The following is an example usage assuming an existing Spark DataFrame split into `training` and `test`:

```python
from pipeline_tuning import DagCrossValidator


tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

paramGrid = ParamGridBuilder() \
    .addGrid(hashingTF.numFeatures, [10, 100, 1000]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

crossval = DagCrossValidator(estimator=pipeline,
                             estimatorParamMaps=paramGrid,
                             evaluator=BinaryClassificationEvaluator(),
                             parallelism=2)

cvModel = crossval.fit(training)
cvModel.transform(test).show()
```
