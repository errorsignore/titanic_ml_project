from .nodes import fit_best_model
from kedro.pipeline import Pipeline, node, pipeline

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=fit_best_model,
                inputs=[
                    "params:models",
                    "X_train",
                    "X_test",
                    "y_train",
                    "y_test",
                    "params:SEED",
                    "params:proba_threshold"
                    ],
                outputs=[
                    "titanic_model_classifier",
                    "model_metrics",
                    "feature_importance_plot"
                    ],
                name="Fitting_best_model",
            ),
        ],
        namespace="data_science",
        inputs=[
            "X_train",
            "X_test",
            "y_train",
            "y_test"
            ],
        outputs=[
            "titanic_model_classifier",
            "model_metrics",
            "feature_importance_plot"
            ],
    )
