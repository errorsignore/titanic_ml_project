import pandas as pd
from titanic_ml_project.pipelines.data_engineering.nodes import (
    age_binning
    , fare_binning
)

class MLPredictor:
    def __init__(self, model):
        self.model = model

    def predict(self, args_API: pd.DataFrame):
        df_args = args_API

        df_args = age_binning(df_args)
        df_args = fare_binning(df_args)
        df_args['FamilySize'] = (
            df_args['SibSp'] + df_args['Parch'] + 1
        )
        df_args['isAlone'] = (
            df_args['FamilySize']
            .apply(
                lambda x: 1 if x==1 else 0
            )
        )
        df_args = df_args.astype(
            {'hasCabin': 'int64'}
        )
        
        survival_rate_probability = (
            self.model.predict_proba(df_args)[0][1]
        )
        return {
            "prediction": survival_rate_probability
        }


def save_predictor(model):
    predictor = MLPredictor(model)

    if testing_prediction_function(predictor):
        return predictor

    raise "Something went wrong"

def testing_prediction_function(
    predictor
) -> bool:
    test_data = {
        'Pclass': [3],
        'Age': [0.0],
        'SibSp': [0],
        'Parch': [2],
        'Fare': [2.0],
        'FamilySize': [3],
        'hasCabin': [0],
        'isAlone': [0],
        'Sex': ['female'],
        'Embarked': ['C'],
        'Title': ['Miss']
    }
    test_data = pd.DataFrame(test_data)

    prediction = predictor.predict(test_data)
    print(prediction['prediction'])
    return True