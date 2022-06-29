"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from titanic_ml_project.pipelines import pre_processing as pp
from titanic_ml_project.pipelines import data_engineering as de
from titanic_ml_project.pipelines import training as train


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    pre_processing_pipeline = pp.create_pipeline()
    data_engineering_pipeline = de.create_pipeline()
    training_pipeline = train.create_pipeline()

    return {
        "__default__":
            pre_processing_pipeline
            + data_engineering_pipeline
            + training_pipeline,
        "pp": pre_processing_pipeline,
        "de": data_engineering_pipeline,
        "training": training_pipeline,
    }
