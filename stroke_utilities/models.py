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
    """
    Calculate the mean absolute value per feature for some SHAP values.
    
    Inputs:
    -------
    shap_values - n patients x n features array. Output from 
                  explainer.shap_values(data). 
    features    - list or array. List of feature names in the same
                  order as the shap_values index.

    Returns:
    --------
    mean_abs_shap - pandas Series. The mean shap values for each
                    feature averaged over all patients, and sorted
                    in descending order.
    """
    mean_abs_shap = pd.Series(
        np.mean(np.abs(shap_values), axis=0),
        index=features)
    mean_abs_shap.sort_values(inplace=True, ascending=False)
    return mean_abs_shap


def calculate_mean_absolute_shap_values_LGBM(shap_values, features):
    """
    Calculate the mean absolute value per feature for some SHAP values.
    
    Inputs:
    -------
    shap_values - list of two arrays. Output from 
                  explainer.shap_values(data). Each array contains
                  one array per patient, and each patient array
                  has as many values as are in "features".
    features    - list or array. List of feature names in the same
                  order as the shap_values index.

    Returns:
    --------
    mean_abs_shap - pandas Series. The mean shap values for each
                    feature averaged over all patients, and sorted
                    in descending order.
    """
    mean_abs_shap = pd.Series(
        np.mean(np.abs(shap_values[1]), axis=0),
        index=features)
    mean_abs_shap.sort_values(inplace=True, ascending=False)
    return mean_abs_shap


def calculate_shap_values_by_category_LGBM(data, category, shap_values, features):
    """
    Group the SHAP values by a category and average across patients.

    Inputs:
    -------
    data        - pandas DataFrame. "X" data of everything but the 
                  predicted values. Column names should match those
                  in "features" list.
    category    - str. Name of the feature to group the results by.
    shap_values - list of two arrays. Output from 
                  explainer.shap_values(data). Each array contains
                  one array per patient, and each patient array
                  has as many values as are in "features".
    features    - list. List of features in the shap values.

    Returns:
    --------
    team_shap - pandas DataFrame. SHAP value averages across all 
                patients. The mean and std are given here.
    """
    # Find where this category is in the shap values list:
    ind_stroke_team = features.index(category)
    
    # DataFrame of hopsital SHAP values for all patients
    all_team_shap = pd.DataFrame()
    all_team_shap[category] = data[category]
    all_team_shap['SHAP'] = shap_values[1][:, ind_stroke_team]

    all_teams = sorted(list(set(all_team_shap[category])))
    all_means = []
    all_medians = []
    all_stds = []
    for team in all_teams:
        inds = np.where(all_team_shap[category] == team)[0]
        t_median = np.nanmedian(all_team_shap['SHAP'][inds])
        t_mean = np.nanmean(all_team_shap['SHAP'][inds])
        t_std = np.nanstd(all_team_shap['SHAP'][inds])

        all_medians.append(t_median)
        all_means.append(t_mean)
        all_stds.append(t_std)
        

    team_shap = pd.DataFrame()
    team_shap[category] = all_teams
    team_shap['SHAP_median'] = all_medians
    team_shap['SHAP_mean'] = all_means
    team_shap['SHAP_std'] = all_stds
    
    # # Get summary for teams
    # team_shap = pd.DataFrame()
    # a = all_team_shap.groupby(category)#.agg({'SHAP': lambda x: x.mean(skipna=False)})
    # team_shap[category] = a.groups.keys()
    # team_shap['SHAP_mean'] = a.mean()
    # team_shap['SHAP_median'] = a.median()
    # team_shap['SHAP_std'] = a.std()
    return team_shap


def select_top_values_in_dataframe(df, column, n_to_keep, ascending=False):
    """
    Sort dataframe by a column and keep only the top few values.

    Inputs:
    -------
    df        - pandas DataFrame. The data to be sorted.
    column    - str. The column to sort by.
    n_to_keep - int. How many of the top values to keep.
    ascending - bool. Whether to sort by in/de-creasing values.

    Returns:
    --------
    The sorted and truncated dataframe.
    """
    df = df.sort_values(column, ascending=ascending)
    return df[:n_to_keep]