import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import yfinance as yf
from datetime import date 
from pandas_datareader import data as pdr


class CreateDataFrame(object):
    '''
    This class is used to create the rolling average on all features and
    targets in the input dataframe. This is necessary because stock data
    will usually have a upward trend that greatly impacts the error in 
    a model like LSTM or Random Forest.

    Parameters:
    The dataframe
    The days back for rolling average

    Returns:
    A new dataframe with rolling averages minus true values, and true values
    '''

    def __init__(self, ticker, data, days_back):
        self.ticker = ticker
        self.df = data
        self.days_back = days_back
    
    def new_features(self):
        self.df.columns = map(str.lower, self.df.columns)
        self.df.columns = self.df.columns.str.replace(' ', '_')
        self.df['tmr_high'] = self.df['high'].shift(periods=-1)
        self.df['tmr_low'] = self.df['low'].shift(periods=-1)
        self.df['tmr_high'].fillna(self.df['high'], inplace=True)
        self.df['tmr_low'].fillna(self.df['low'], inplace=True)
        self.df['avg_price'] = self.df[['high', 'low', 'open', 'close']].sum(axis=1)/4
        cols = list(self.df.columns)
        # Create all rolling averages for all columns in the dataframe
        for col in cols:
            self.df['rolling_avg_' + col] = self.df[col].rolling(self.days_back, center=True).mean()
        self.df = self.df.dropna(axis=0, how='any')
        # Prepare the feature column lists
        cols_t = [cols.pop(cols.index('tmr_high')), cols.pop(cols.index('tmr_low'))]
        rolling_avg_feature_cols = []
        feature_cols = []
        # Prepare the target column lists
        rolling_avg_target_cols = ['rolling_avg_tmr_high', 'rolling_avg_tmr_low']
        target_cols = ['target_tmr_high', 'target_tmr_low']
        for col in cols:
            rolling_avg_feature_cols.append('rolling_avg_' + col)
            feature_cols.append('feature_' + col)
        # Create the features (cols) by feature minus associated rolling avg, concat with df
        for i in range(len(cols)):
            feature_cols[i] = pd.Series(self.df.apply(lambda row: row[rolling_avg_feature_cols[i]] - row[cols[i]], axis=1), name=feature_cols[i])
            self.df = pd.concat([self.df, feature_cols[i]], axis=1)
        # Create the targets (cols_targets) by target minus associated rolling avg, concat with df
        for i in range(len(target_cols)):
            target_cols[i] = pd.Series(self.df.apply(lambda row: row[rolling_avg_target_cols[i]] - row[cols_t[i]], axis=1), name=target_cols[i])
            self.df = pd.concat([self.df, target_cols[i]], axis=1)
        self.df.to_csv(f'../data/{self.ticker}_prepared_dataframe.csv')
        return self.df

if __name__ == '__main__':
    # Import the stock data
    start_date = '2000-01-01'
    end_date = date.today()
    ticker = 'SPY'
    data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
    CreateDataFrame(ticker, data, 5).new_features()
    