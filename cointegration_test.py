import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

class CointegrationPair:

    
    def __init__(self, time_series_1, time_series_2):
        self.np_time_series_1 = np.genfromtxt(time_series_1, delimiter=',', skip_header = 1)
        self.np_time_series_2 = np.genfromtxt(time_series_2, delimiter=',', skip_header = 1)
        self.np_time_series_1_returns = np.empty(0)
        self.np_time_series_2_returns = np.empty(0)
        self.relationship = 0
        self.stationary_distr = np.empty(0)
    
    def generate_log_returns(self):
        for index in range(len(self.np_time_series_1) - 1):
            self.np_time_series_1_returns = np.append(self.np_time_series_1_returns, (self.np_time_series_1[index + 1] - self.np_time_series_1[index]))
        
        for index in range(len(self.np_time_series_2) - 1):
            self.np_time_series_2_returns = np.append(self.np_time_series_2_returns, (self.np_time_series_2[index + 1] - self.np_time_series_2[index]))
        
    def generate_linear_relationship(self):
        classifier = LinearRegression(n_jobs = -1)
        classifier.fit(self.np_time_series_1_returns.reshape(-1,1), self.np_time_series_2_returns.reshape(-1,1))
        
        self.relationship = classifier.coef_
    
    def create_stationary_distr(self):
        stationary_distr = self.np_time_series_2_returns - (self.relationship * self.np_time_series_1_returns)        
        self.stationary_distr = stationary_distr
    
    def test_stationarity(self):
        self.generate_log_returns()
        self.generate_linear_relationship()
        self.create_stationary_distr()
        

        adf_result = adfuller(self.stationary_distr[0])
        if adf_result[1] < 0.01:
            return True, adf_result[1]
        else:
            return False
        
    def plot(self):
        plt.plot(self.stationary_distr[0])
        plt.xlabel("Date")
        plt.ylabel("Diff in Log Returns")
        plt.show()
        

    
