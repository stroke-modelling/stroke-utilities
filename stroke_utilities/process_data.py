"""
Functions for processing data.

"""

def restrict_data_to_range(df, low, high, column):
    try:
        mask = (df[column] >= low) & (df[column] <= high)
        df = df[mask]
        df.drop(column, inplace=True, axis=1)
    except KeyError:
        # "year" isn't in the input data.
        pass
    return df


def split_X_and_y(df, column):
    X = df.drop(column, axis=1)
    y = df[column]
    return X, y