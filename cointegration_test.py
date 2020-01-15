import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

    
def gen_log_returns(np_time_series_1, np_time_series_2):
    np_time_series_1_returns = np.empty(0)
    np_time_series_2_returns = np.empty(0)
    for index in range(len(np_time_series_1) - 1):
        np_time_series_1_returns = np.append(np_time_series_1_returns, (np_time_series_1[index + 1] - np_time_series_1[index]))
    
    for index in range(len(np_time_series_2) - 1):
        np_time_series_2_returns = np.append(np_time_series_2_returns, (np_time_series_2[index + 1] - np_time_series_2[index]))
            
    return np_time_series_1_returns, np_time_series_2_returns
        
def gen_linear_relationship(np_time_series_1_returns, np_time_series_2_returns):
    classifier = LinearRegression(n_jobs = -1)
    classifier.fit(np_time_series_1_returns.reshape(-1,1), np_time_series_2_returns.reshape(-1,1))
        
    return classifier.coef_
    
def gen_stationary_distr(np_time_series_1_returns, np_time_series_2_returns, coefficient):
    stationary_distr = np_time_series_2_returns - (coefficient * np_time_series_1_returns)        
    return stationary_distr
    
def test_stationarity(np_time_series_1, np_time_series_2):
    np_time_series_1_returns, np_time_series_2_returns = gen_log_returns(np_time_series_1, np_time_series_2)
    coefficient = gen_linear_relationship(np_time_series_1_returns, np_time_series_2_returns)
    stationary_distr = gen_stationary_distr(np_time_series_1_returns, np_time_series_2_returns, coefficient)
        

    adf_result = adfuller(stationary_distr[0])
    if adf_result[1] < 0.01:
        return True, adf_result[1]
    else:
        return False, adf_result[1]
        
def plot(stationary_distr):
    plt.plot(stationary_distr[0])
    plt.xlabel("Time")
    plt.ylabel("Diff in Log Returns")
    plt.show()
    

        

    
