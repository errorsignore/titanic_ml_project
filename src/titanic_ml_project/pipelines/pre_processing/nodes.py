import pandas as pd

def fill_missing_age(
    titanic_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Node for filling missing age.
    """
    titanic_data['Age'] = (
        titanic_data['Age']
        .fillna(
            titanic_data['Age']
            .median()
        )
    )
    
    return titanic_data
