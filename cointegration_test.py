import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

#currently unused. may be implemented in the future
'''def __gen_data_list(ticker_list, file_path):
    ticker_data_dict = {}
    for ticker in ticker_list:
        ticker_data_dict.update({ticker : np.genfromtxt(file_path + ticker + '.csv', delimiter=',', skip_header=1)})
    
    return ticker_data_dict
'''
 
def gen_log_returns(np_time_series):
    np_time_series_returns = np.empty(0)
    for index in range(len(np_time_series) - 1):
        np_time_series_returns = np.append(np_time_series_returns, (np_time_series[index + 1] - np_time_series[index]))
           
    return np_time_series_returns
        
def gen_linear_relationship(np_time_series_1_returns, np_time_series_2_returns):
    classifier = LinearRegression(n_jobs = -1)
    classifier.fit(np_time_series_1_returns.reshape(-1,1), np_time_series_2_returns.reshape(-1,1))
        
    return classifier.coef_
    
def gen_stationary_distr(np_time_series_1_returns, np_time_series_2_returns, coefficient):
    stationary_distr = np_time_series_2_returns - (coefficient * np_time_series_1_returns)        
    return stationary_distr
    
def test_stationarity(np_time_series_1, np_time_series_2):
    np_time_series_1_returns = gen_log_returns(np_time_series_1)
    np_time_series_2_returns = gen_log_returns(np_time_series_2)
    coefficient = gen_linear_relationship(np_time_series_1_returns, np_time_series_2_returns)
    stationary_distr = gen_stationary_distr(np_time_series_1_returns, np_time_series_2_returns, coefficient)
        

    adf_result = adfuller(stationary_distr[0])
    if adf_result[1] < 0.01:
        null_hyp = "Rejected null hypothesis"
        return null_hyp, adf_result[1]
    else:
        null_hyp = "Failed to Reject"
        return null_hyp, adf_result[1]
    
def find_pairs(ticker_data_dict):
    ticker_pair_list = combinations(ticker_data_dict, 2)
    if len(ticker_pair_list) == 0:
        raise Exception("Your dictionary had either 0 or 1 entries. No pairs can be made")
    results = []
    try:
        for pair in ticker_pair_list:
            time_series_1 = ticker_data_dict[pair[0]]
            time_series_2 = ticker_data_dict[pair[1]]
            
            null_hyp, p_val = test_stationarity(time_series_1, time_series_2)
            pair_results = [pair, null_hyp, p_val]
            results.append(pair_results)
    except IndexError:
        raise Exception("Index Error. Make sure you passed a dictionary with tickers as keys and time series as values")
    
    return results
        
def plot(stationary_distr):
    plt.plot(stationary_distr[0])
    plt.xlabel("Time")
    plt.ylabel("Diff in Log Returns")
    plt.show()
    

        

    
