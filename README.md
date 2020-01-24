# This package is deprecated. Up to date code can be found under StatArbTools

# Cointegration_Test

cointegration_test is a Python library primarily for determining if a pair of time series are cointegrated.
It also includes tools for generating an array of log returns from a price array, looking for a linear relationship,
and creating a potentially stationary distribution.


## Usage

```python
import cointegration_test

cointegration_test.gen_log_returns(numpy_time_series_1, numpy_time_series_2) # returns numpy arrays of the log returns for each time series
cointegration_test.gen_linear_relationship(numpy_log_returns_1, numpy_log_returns_2) # returns the coefficient from a linear regression between the two log returns arrays
cointegration_test.gen_stationary_distr(numpy_log_returns_1, numpy_log_returns_2, coefficient) # returns the linear combination of the two log returns arrays based on a linear regression coefficient
cointegration_test.test_stationarity(numpy_time_series_1, numpy_time_series_2) # returns True if the null hypothesis of an Augmented Dickey Fuller test is rejected and False otherwise. It also returns the p-value of the ADF test.
cointegration_test.plot(stationary_distribution) # plots the passed distribution
```

## Contributing
For changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
