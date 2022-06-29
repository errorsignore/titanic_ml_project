from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    create_title
    , create_family_column
    , fare_binning
    , age_binning
    , has_cabin_binary
    , is_alone
    , create_model_input
    , split_data
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=create_title,
                inputs= "pre_processed_titanic_data",
                outputs="create_title_titanic_data",
                name="Create_title_based_on_Name",
            ),
            node(
                func=create_family_column,
                inputs= "create_title_titanic_data",
                outputs="create_family_titanic_data",
                name="Create_family_size_column",
            ),
            node(
                func=fare_binning,
                inputs= "create_family_titanic_data",
                outputs="fare_binning_titanic_data",
                name="Fare_binning_column",
            ),
            node(
                func=age_binning,
                inputs= "fare_binning_titanic_data",
                outputs="age_binning_titanic_data",
                name="Age_binning_column",
            ),
            node(
                func=has_cabin_binary,
                inputs= "age_binning_titanic_data",
                outputs="has_cabin_titanic_data",
                name="Create_binary_on_cabin_column",
            ),
            node(
                func=is_alone,
                inputs= "has_cabin_titanic_data",
                outputs="is_alone_titanic_data",
                name="is_alone_titanic_data",
            ),
            node(
                func=create_model_input,
                inputs= [
                    "is_alone_titanic_data",
                    "params:columns_to_keep"
                    ],
                outputs="model_input_titanic_data",
                name="Create_model_input",
            ),
            node(
                func=split_data,
                inputs= [
                    "model_input_titanic_data"
                    , "params:target_column"
                    , "params:train_test_ratio"
                    , "params:SEED"
                ],
                outputs=[
                    "X_train"
                    , "X_test"
                    , "y_train"
                    , "y_test"
                ],
                name="Split_Data",
            ),
        ],
        namespace="data_engineering",
        inputs= "pre_processed_titanic_data",
        outputs=[
            "X_train"
            , "X_test"
            , "y_train"
            , "y_test"
            ],
    )
