from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    fill_missing_age
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fill_missing_age,
                inputs= "titanic_data",
                outputs="pre_processed_titanic_data",
                name="Fill_Missing_Age",
            ),
        ],
        namespace="data_preprocessing",
        inputs= "titanic_data",
        outputs="pre_processed_titanic_data",
    )
