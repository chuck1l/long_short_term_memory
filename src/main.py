import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import yfinance as yf
from datetime import date 
from feature_engineering import CreateDataFrame
from pandas_datareader import data as pdr

# Import the stock data
start_date = '2000-01-01'
end_date = date.today()
ticker = 'SPY'
data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
prep_data = CreateDataFrame(ticker, data, 5).new_features()
print(prep_data.head())