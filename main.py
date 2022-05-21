from typing import List

import pandas as pd
import numpy as np
import scipy.stats as st
import user_input
import query_data
import numerical_functions

# globals / inputs
ewma_lambda: float = .95
confidence_level: float
holding_period: int
start_date: str
end_date: str


def get_ew_return(ewma_lambda, prev, current_price, prev_price) -> float:
    return np.sqrt(ewma_lambda * np.square(prev) + (1 - ewma_lambda) * np.square(np.log(current_price / prev_price)))


def get_volatility_estimate(ewma_lambda, prices, averaging_type) -> float:
    if averaging_type == "simple":
        return np.average(np.abs(np.log(prices).diff()[1:]))
    elif averaging_type == "ewma":
        vol_estimate = np.log(prices[1] / prices[0])
        for i in range(1, len(prices) - 1):
            vol_estimate = get_ew_return(ewma_lambda, vol_estimate, prices[i + 1], prices[i])
        return vol_estimate
    else:
        raise ValueError("Bad averaging type passed to get_volatility_estimate")


def my_covariance(a, b):  # sample covariance
    if len(a) != len(b):
        raise ValueError("arrays not same size")

    z = 0
    for x, y in zip(a, b):
        z += (x - np.average(a)) * (y - np.average(b))

    return z / (len(a) - 1)


def my_corrcoef(a, b):  # sample correlation coefficient
    return my_covariance(a, b) / (np.sqrt(np.cov(a)) * np.sqrt(np.cov(b)))


tickers, positions, start_date, end_date, confidence_level, holding_period = user_input.get_arguments()

# main body of code

returns_list: List[List[float]] = []
prices_list: List[List[float]] = []
forwards_list: List[float] = []

for ticker in tickers:
    prices = query_data.get_prices(ticker, start_date, end_date, 'Close')
    returns = np.log(prices).diff()[1:]
    returns_list.append(returns)
    prices_list.append(prices)
    forwards_list.append(list(prices).pop())

df = pd.DataFrame(returns_list).transpose()
df.columns = tickers
corr_mx = numerical_functions.get_corr_mx(df, "ewma", tickers, ewma_lambda)

vol_list = []
for price_series in prices_list:
    vol_list.append(get_volatility_estimate(ewma_lambda, price_series, "ewma"))
vol_mx = np.identity(len(vol_list))
np.fill_diagonal(vol_mx, np.array(vol_list))

vcv_mx = (vol_mx.dot(corr_mx)).dot(vol_mx)

forwards = np.array(forwards_list)
notional_values = positions * forwards
weights = notional_values / np.sum(notional_values)

final = float((weights.dot(vcv_mx)).dot(weights.transpose()))
portfolio_stddev = np.sqrt(final)
z_score = st.norm.ppf(confidence_level)
shift = np.expm1(portfolio_stddev * z_score) * np.sqrt(holding_period)

var = abs(np.sum(notional_values) * shift)
print("$" + f'{var:,.2f}')
