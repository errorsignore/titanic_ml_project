import pandas as pd
from sklearn.model_selection import train_test_split

from titanic_ml_project.helpers.helper_functions import (
    get_title
)

def create_title(
    titanic_data: pd.DataFrame
) -> pd.DataFrame:
    """
    
    """
    titanic_data['Title'] = (
        titanic_data['Name']
        .apply(
            get_title
        )
    )

    titanic_data['Title'] = (
        titanic_data['Title']
        .replace(
            ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr',
            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
            'Rare')
    )

    titanic_data['Title'] = (
        titanic_data['Title']
        .replace('Mlle', 'Miss')
    )
    titanic_data['Title'] = (
        titanic_data['Title']
        .replace('Ms', 'Miss')
    )
    titanic_data['Title'] = (
        titanic_data['Title']
        .replace('Mme', 'Mrs')
    )

    return titanic_data

def create_family_column(
    titanic_data: pd.DataFrame
) -> pd.DataFrame:
    """
    
    """
    titanic_data['FamilySize'] = (
        titanic_data['SibSp'] + titanic_data['Parch'] + 1
    )

    return titanic_data

def fare_binning(
    titanic_data: pd.DataFrame
) -> pd.DataFrame:
    """
    This function bins Fare based on its value.
    """
    titanic_data.loc[titanic_data['Fare'] <= 7.91, 'Fare'] = 0
    titanic_data.loc[(titanic_data['Fare'] > 7.91) & (titanic_data['Fare'] <= 14.454), 'Fare'] = 1
    titanic_data.loc[(titanic_data['Fare'] > 14.454) & (titanic_data['Fare'] <= 31), 'Fare'] = 2
    titanic_data.loc[titanic_data['Fare'] > 31, 'Fare'] = 3

    return titanic_data

def age_binning(
    titanic_data: pd.DataFrame
) -> pd.DataFrame:
    """
    This functions bins Age based on its value.
    """
    titanic_data.loc[titanic_data['Age'] <= 16, 'Age'] = 0
    titanic_data.loc[(titanic_data['Age'] > 16) & (titanic_data['Age'] <= 32), 'Age'] = 1
    titanic_data.loc[(titanic_data['Age'] > 32) & (titanic_data['Age'] <= 48), 'Age'] = 2
    titanic_data.loc[(titanic_data['Age'] > 48) & (titanic_data['Age'] <= 64), 'Age'] = 3
    titanic_data.loc[titanic_data['Age'] > 64, 'Age'] = 4

    return titanic_data

def has_cabin_binary(
    titanic_data: pd.DataFrame
) -> pd.DataFrame:
    """
    As Cabin is a column with many empty values, we create a feature
    to tell if the passanger had a Cabin.
    """
    titanic_data['hasCabin'] = (
        titanic_data['Cabin']
        .apply(
            lambda x: 1 if x == x else 0
        )
    )
    return titanic_data

def is_alone(
    titanic_data: pd.DataFrame
) -> pd.DataFrame:
    """
    This function checkes whether the
    passanger in the boat is alone or not.
    """
    titanic_data['isAlone'] = (
        titanic_data['FamilySize'].apply(
            lambda x: 1 if x == 1 else 0
        )
    )
    
    return titanic_data

def create_model_input(
    titanic_data: pd.DataFrame,
    columns_to_keep: list
) -> pd.DataFrame:
    """
    Create Model Input.
    """
    titanic_data = titanic_data[columns_to_keep]

    numeric_cols = (
        titanic_data
        .select_dtypes(
            include='number'
        ).columns.tolist()
    )
    object_cols = (
        titanic_data
        .select_dtypes(
            include='object'
        ).columns.tolist()
    )

    titanic_data = (
        titanic_data[numeric_cols + object_cols]
    )

    return titanic_data

def split_data(
    titanic_data: pd.DataFrame,
    target_col: str,
    train_test_ratio: float,
    SEED: int,
):
    """
    Node for splitting the data into train and test set.
    """

    X = titanic_data.drop([target_col], axis = 1)
    y = titanic_data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = train_test_ratio,
        random_state = SEED, stratify = y
    )

    return X_train, X_test, y_train, y_test
