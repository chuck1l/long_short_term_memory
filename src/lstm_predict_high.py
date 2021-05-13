import math
import numpy as np 
import pandas as pd 
# LSTM imports
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras.models import Sequential

class LstmPredictHigh(object):
    def __init__(self, data):
        self.data = data
        self.data.set_index('Date', inplace=True)

    def column_manipulation(self):
        self.y = self.data['target_tmr_high']
        self.df = self.data.loc[:, self.data.columns.str.startswith('feature')]

    def train_test_holdout(self):
        X, self.holdout = self.df.drop(self.df.tail(1).index), self.df.tail(1)
        self.y = self.y.drop(self.y.tail(1).index)
        train_end = int(X.shape[0]*.67)
        self.X_train = X.iloc[0:train_end, :].copy()
        self.X_test = X.iloc[train_end:, :].copy()
        self.y_train = self.y.iloc[0:train_end].copy()
        self.y_test = self.y.iloc[train_end:].copy()

    def reshape_tr_test(self):
        # Reshape the data for LSTM format
        self.X_train = np.expand_dims(self.X_train.values[:, :], axis=2)
        self.X_test = np.expand_dims(self.X_test.values[:, :], axis=2)
        self.holdout = np.expand_dims(self.holdout, axis=2)
        self.y_train = self.y_train.values.reshape(-1, 1)
        self.y_test = self.y_test.values.reshape(-1, 1)

    def make_predictions(self):
        # Input Dimension
        input_dim = self.X_train.shape[1]
        # Initializing the Neural Network Based On LSTM
        model = Sequential()
        
        model.add(LSTM(units=25, return_sequences=True, input_shape=(input_dim, 1)))
        model.add(Dropout(0.25))

        model.add(LSTM(units=10))
        model.add(Dropout(0.25))
        
        model.add(Dense(units=1, activation='relu'))
        
        model.compile(loss='mean_squared_error', optimizer ='adam')
        
        model.fit(self.X_train, self.y_train, epochs=500, batch_size=10, verbose=1)
        # Save Model 
        #model.save('../models/high_model.h5')
        # make predictions
        self.trainPredict = model.predict(self.X_train)
        self.testPredict = model.predict(self.X_test)
        #self.hoPredict = model.predict(self.holdout)
        # Calculate RMSE
        trainScore = math.sqrt(mean_squared_error(self.y_train, self.trainPredict))
        print('High of Day Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(self.y_test, self.testPredict))
        print('High of Day Test Score: %.2f RMSE' % (testScore))
        # hoScore = math.sqrt(mean_squared_error(self.holdout_df['tmr_high'].values, self.hoPredict))
        # print('High of Day Holdout Score: %.2f RMSE' % (hoScore))

    def get_prediction(self):
        self.column_manipulation()
        self.train_test_holdout()
        self.reshape_tr_test()
        self.make_predictions()

if __name__ == '__main__':
    data = pd.read_csv('../data/SPY_prepared_dataframe.csv')
    LstmPredictHigh(data).get_prediction()