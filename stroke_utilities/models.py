"""
Set up and train machine learning models.
"""
from lightgbm import LGBMClassifier
import shap
import pandas as pd
import numpy as np

def train_LGBMClassifier(X, y, **kwargs):
    """
    kwargs are passed to LGBMClassifier
    """
    # Set default values:
    if 'random_state' not in kwargs.items():
        kwargs['random_state'] = 42
    else:
        pass
    if 'learning_rate' not in kwargs.items():
        kwargs['learning_rate'] = 0.3
    else:
        pass
    
    # Define model
    model = LGBMClassifier(**kwargs)

    # Fit model
    model.fit(X, y)
    return model


def create_shap_TreeExplainer(model):
    """
    Is this worth its existence?
    """
    return shap.TreeExplainer(model)


def calculate_shap_values(explainer, data):
    shap_values_extended = explainer(data)
    shap_values = explainer.shap_values(data)
    return shap_values, shap_values_extended


def calculate_mean_absolute_shap_values(shap_values, features):
    mean_abs_shap = pd.Series(
        np.mean(np.abs(shap_values[1]), axis=0),
        index=features)
    mean_abs_shap.sort_values(inplace=True, ascending=False)
    return mean_abs_shap


def calculate_shap_values_by_category(data, category, shap_values, features):
    # Find where this category is in the shap values list:
    ind_stroke_team = features.index(category)
    
    # DataFrame of hopsital SHAP values for all patients
    all_team_shap = pd.DataFrame()
    all_team_shap[category] = data[category]
    all_team_shap['SHAP'] = shap_values[1][:, ind_stroke_team]

    # Get summary for teams
    team_shap = pd.DataFrame()
    team_shap[category] = all_team_shap.groupby(category).groups.keys()
    team_shap['SHAP_mean'] = all_team_shap.groupby(category).mean()
    team_shap['SHAP_std'] = all_team_shap.groupby(category).std()
    return team_shap


def select_top_values_in_dataframe(df, column, n_to_keep, ascending=False):
    df = df.sort_values(column, ascending=ascending)
    return df[:n_to_keep]