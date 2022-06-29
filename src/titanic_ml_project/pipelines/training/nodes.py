import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict

from sklearn.impute import SimpleImputer

from titanic_ml_project.helpers.helper_functions import (
    model_from_string
)

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.ensemble import VotingClassifier

from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import seaborn as sns
import matplotlib.pyplot as plt

def get_column_transformer():
    """
    This functions creates a Column Transformer for both
      numerical and categorical columns.
    """
    numeric_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(missing_values = np.nan, strategy='constant', fill_value=0)),
            ('encoder', StandardScaler())
        ])

    categorical_transformer = Pipeline(
        steps = [
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing_value')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

    CT_pipeline = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include='number')),
            ("cat", categorical_transformer, make_column_selector(dtype_include='object')),
        ])

    return CT_pipeline

def get_metrics(
    X_test, 
    y_test,
    clf,
    model_name: str,
    th: float
):
    """
    This function gets all the metrics given for a Classifier.
    """
    y_pred_prob = clf.predict_proba(X_test)[:,1]
    y_pred = (y_pred_prob >= th).astype(int)
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    tn, fp, fn, tp  = confusion_matrix(y_test, y_pred).ravel()
    
    return pd.DataFrame({
        'model': [model_name],
        'th': [th],
        'F1': [f1],
        'AUC': [auc],
        'Precision': [precision],
        'Recall': [recall],
        'FPR': [fp/(fp+tn)],
        'TP': [tp],
        'FP': [fp],
        'TN': [tn],
        'FN': [fn],
        'hyperparameters': [clf.get_params()]
    })

def generate_feature_importance_plot(
    feature_name_list: list,
    classifier_list: list,
) -> list:
    """
    This function generates the feature importance plot
    for the given Classifier.
    """
    fig, axes = plt.subplots(len(classifier_list), 1, sharex = True, figsize=(20,20))
    
    axes = axes.ravel()
    
    for index, model_list in enumerate(classifier_list):
        model_name = model_list[0]
        model = model_list[1]
        
        importance_list = [round(elem, 2) for elem in model.feature_importances_  * 100 / sum(model.feature_importances_)]
        
        feat_imp = pd.DataFrame(
            {'column': feature_name_list,
            'importance': importance_list}
        )
        feat_imp.sort_values(by='importance', ascending=False, inplace=True)

        feat_imp = feat_imp.head(15)
                
        plot = sns.barplot(x='importance', y='column', data=feat_imp, color='b', ax=axes[index])
        for i in plot.containers:
            plot.bar_label(i,)
            
        plot.set_title(model_name)
        plot.set(xlabel=None, ylabel=None)
        
        del feat_imp
            
    return plot.get_figure()

def get_best_model(
    model: dict,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    SEED: int
):
    """
    This function gets a model with a set of parameters and return the best one based on F1 score.
    Input:
      Model: A classifier object with variable parameters.
      X_train: The model training data.
      y_train: The model training prediction data.
      SEED: A constant number to get the same results of Folding.
    Output:
      Best Model Classifier.
    """
    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )
    
    clf = model_from_string(model['classifier'])

    clf_grid = GridSearchCV(
            estimator = clf,
            param_grid = model['params'],
            cv = kfold,
            scoring = 'f1',
            verbose = 2
            )
    
    clf_grid.fit(X_train, y_train)

    return clf_grid.best_estimator_

def fit_best_model(
    models:dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    SEED: int,
    th: float,
):
    """
    This functions fits the best model.
    First, it creates a column transformer object;
    Second, we find the model with the best parameters of each type in a GridSearch;
    Third, we get all the models into a VotingClassifier, fit it and get the metrics;
    Lastly, we get the feature importance plot to see which variables influence more in the result.
    """
    # Transforming Data
    CT_pipeline = get_column_transformer()

    # Get column names out
    CT_pipeline.transformers[1][1].fit(X_train.select_dtypes(include='object'))
    column_names = (
        X_train.select_dtypes(include='number').columns.tolist()
        + CT_pipeline.transformers[1][1].get_feature_names_out().tolist()
    )

    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()

    X_train = CT_pipeline.fit_transform(X_train)

    X_test = CT_pipeline.transform(X_test)

    metrics_df = pd.DataFrame()

    clf_list = []

    for model in models:
        clf = get_best_model(model, X_train, y_train, SEED)
        clf_list.append(
            (model['name'], clf)
        )
        metrics_df = pd.concat([
            metrics_df,
            get_metrics(X_test, y_test, clf, model['name'] + '_test', th)
        ])
        metrics_df = pd.concat([
            metrics_df,
            get_metrics(X_train, y_train, clf, model['name'] + '_train', th)
        ])
    
    clf = VotingClassifier(estimators = clf_list, voting='soft')

    clf.fit(X_train, y_train)

    metrics_df = pd.concat([
        metrics_df,
        get_metrics(X_test, y_test, clf, 'voting_clf_test', th)
    ])
    metrics_df = pd.concat([
        metrics_df,
        get_metrics(X_train, y_train, clf, 'voting_clf_train', th)
    ])

    feature_importance_plot = generate_feature_importance_plot(column_names, clf_list)

    run_name = datetime.now().isoformat()

    metrics_df['run_name'] = run_name

    final_pipeline = Pipeline(
        steps = [
            ('CT', CT_pipeline),
            ('classifier', clf)
    ])

    return final_pipeline, metrics_df, feature_importance_plot
