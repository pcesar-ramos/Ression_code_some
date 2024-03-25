# import libraries
import pandas as pd


def make_target_onsets(df: pd.DataFrame, shifters: dict, target: str):
    """
    Generate shifted variables and calculate the maximum for a single ISO code.

    Args:
        df (pd.DataFrame): The input DataFrame assumed to contain data for a single ISO code.
        shifters (dict): A dictionary specifying the shifters.
            Example: {3: 'w3_', 6: 'w6_'} for 2 shifters with prefixes 'w3_' and 'w6_'.
        target (str): The column name of the target variable.

    Returns:
        pd.DataFrame: The modified DataFrame with the maximum for each shifter.
    """
    for shifter, prefix in shifters.items():
        # Loop through each period and generate the shift variables
        for i in range(1, shifter + 1):
            col_name = f'{prefix}{target}{i}'
            df[col_name] = df[target].shift(-i)

        # Take the maximum for t periods forward and create the new variable
        max_col_name = f'recession_ons_{prefix}'
        df[max_col_name] = df[[f'{prefix}{target}{i}' for i in range(1, shifter + 1)]].max(axis=1, skipna=False)

        # Drop the shift variables
        df = df.drop(columns=[f'{prefix}{target}{i}' for i in range(1, shifter + 1)])

        # Shift the resulting column by 1 so it can be used for prediction in a time series classifier
        df[max_col_name] = df[max_col_name].shift(1)

    return df


def create_past_columns_mean(df_, value, windows):
    # counts instances of the target=1 in the past 6, 12, 60, and 120 months
    lcols = (df_[value] # for each country
            .transform(lambda x: x.rolling(w, min_periods=1).mean()) # rolling mean of target
            .fillna(0).rename(f'past{w}_{value}')
            for w in windows) 
    lcols_df = pd.DataFrame(lcols).transpose()
    return df_.join(lcols_df), lcols_df.columns.tolist()




from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import scale

def transform_features(df, period_col='period'):
    """
    Transform features in the DataFrame based on the Augmented Dickey-Fuller (ADF) test.
    If the p-value of a feature is > 0.05, it transforms the variable by taking the percentage change
    over 12 periods, scales it by 100, and drops the original variable. Otherwise, keeps the variable.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'period' column and other columns to be tested and transformed.
        period_col (str): Name of the column that contains the datetime period.
    
    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    transformed_df = df.copy()
    feature_columns = df.columns.drop(period_col)  # Exclude the period column from the features to be transformed
    
    for col in feature_columns:
        adf_result = adfuller(df[col])
        p_value = adf_result[1]
        
        # If p-value > 0.05, transform the variable
        if p_value > 0.05:
            #transformed_col = df[col].pct_change(periods=1) * 100
            transformed_col = df[col].pct_change(periods=12) * 100
            # Scale the transformed column
            # Assuming the intent is to standardize (mean=0, std=1), which is common but let's keep as scale by 100
            transformed_col = (transformed_col - transformed_col.mean()) / transformed_col.std()
            transformed_df[col] = transformed_col.dropna()  # Drop NaN introduced by pct_change

    # Drop any rows with NaN values that may have been introduced by the pct_change operation
    #transformed_df.dropna(inplace=True)
    return transformed_df