from kedro.pipeline import Pipeline, node

from .nodes import save_predictor


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=save_predictor,
                inputs="titanic_model_classifier",
                outputs="MLPredictor",
                name="save_predictor",
            )
        ]
    )
