import pandas as pd

def impute_missing_values(df, transformation_log):
    """
    Impute missing values in numeric columns using the column mean.
    Log the transformation step.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)
            transformation_log.append(
                f"df['{col}'].fillna(df['{col}'].mean(), inplace=True)  # Imputed missing values in '{col}'"
            )
    return df

def drop_duplicates(df, transformation_log):
    """
    Drop duplicate rows from the DataFrame.
    Log the transformation step.
    """
    initial_shape = df.shape
    df.drop_duplicates(inplace=True)
    transformation_log.append(
        f"df.drop_duplicates(inplace=True)  # Dropped duplicates. Shape from {initial_shape} to {df.shape}"
    )
    return df
