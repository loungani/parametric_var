from typing import List

import numpy as np
import pandas as pd
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
specified_column: str = "Close"


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

forwards_list: List[float] = []

first = True
for ticker in tickers:
    prices = query_data.get_prices(ticker, start_date, end_date, 'Close')
    returns = np.log(prices).diff()[1:]

    if first:
        prices_df = prices.to_frame().rename(columns={specified_column: ticker})
        returns_df = returns.to_frame().rename(columns={specified_column: ticker})
        first = False
    else:
        prices_df = prices_df.join(prices.to_frame().rename(columns={specified_column: ticker}))
        returns_df = returns_df.join(returns.to_frame().rename(columns={specified_column: ticker}))

    forwards_list.append(list(prices).pop())

corr_mx = numerical_functions.get_corr_mx(returns_df, "ewma", tickers, ewma_lambda)

vol_list = []
for column in prices_df:
    vol_list.append(numerical_functions.get_volatility_estimate(ewma_lambda, prices_df[column], "ewma"))
vol_mx = np.identity(len(vol_list))
np.fill_diagonal(vol_mx, np.array(vol_list))
vol_mx = pd.DataFrame(vol_mx, columns=tickers, index=tickers)

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
