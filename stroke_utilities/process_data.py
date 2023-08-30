"""
Functions for processing data.

"""
import pandas as pd

def restrict_data_to_range(df, low, high, column):
    """
    Restrict the dataframe to values within a given range.
    
    Inputs:
    -------
    df     - pd.DataFrame. Data to be edited.
    low    - float. Lowest value to keep.
    high   - float. Highest value to keep.
    column - Name of column to restrict values over.

    Returns:
    The edited dataframe.
    """
    try:
        mask = (df[column] >= low) & (df[column] <= high)
        df = df[mask]
        df.drop(column, inplace=True, axis=1)
    except KeyError:
        # "year" isn't in the input data.
        pass
    return df


def split_X_and_y(df, column):
    """
    Split a dataframe into X and y for prediction models.

    Inputs:
    -------
    df     - pandas Dataframe. Input data to be split. 
    column - str. Name of the column to become "y".

    Returns:
    --------
    X - pandas DataFrame. All of the input data except the
        prediction column.
    y - pandas Series. The prediction column of the input data.
    """
    X = df.drop(column, axis=1)
    y = df[column]
    return X, y


def one_hot_encode_column(X, col, prefix='team'):
    # Keep copy of original, with 'Stroke team' not one-hot encoded
    X_combined = X.copy(deep=True)
    
    # One-hot encode 'Stroke team'
    X_hosp = pd.get_dummies(X[col], prefix=prefix)
    X = pd.concat([X, X_hosp], axis=1)
    X.drop(col, axis=1, inplace=True)
    return X
