from typing import List

import pandas as pd
import yfinance as yf
import numpy as np
import scipy.stats as st

# globals / inputs
tickers: List[str] = ['MSFT', 'AAPL', 'GOOGL']
ewma_lambda: float = .95
confidence_level: float = .95
holding_period: int = 5
start_date: str = '2020-04-24'
end_date: str = '2021-04-24'
positions = np.array([50, 100, 10])


def get_prices(ticker, start_date, end_date, column):
    return yf.download(ticker, start=start_date, end=end_date, progress=False)[column]


def get_ew_return(ewma_lambda, prev, current_price, prev_price) -> float:
    return np.sqrt(ewma_lambda * np.square(prev) + (1 - ewma_lambda) * np.square(np.log(current_price / prev_price)))


def get_volatility_estimate(ewma_lambda, prices) -> float:
    vol_estimate = np.log(prices[1] / prices[0])
    for i in range(1, len(prices) - 1):
        vol_estimate = get_ew_return(ewma_lambda, vol_estimate, prices[i + 1], prices[i])
    return vol_estimate


def my_covariance(a, b):  # sample covariance
    if len(a) != len(b):
        raise ValueError("arrays not same size")

    z = 0
    for x, y in zip(a, b):
        z += (x - np.average(a)) * (y - np.average(b))

    return z / (len(a) - 1)


def my_corrcoef(a, b):  # sample correlation coefficient
    return my_covariance(a, b) / (np.sqrt(np.cov(a)) * np.sqrt(np.cov(b)))


def create_ewma_weights(length, ewma_lambda):
    mx = np.ones(length)
    mx[0] = 1 - ewma_lambda
    for i in range(1, len(mx)):
        mx[i] = mx[i - 1] * ewma_lambda
    return mx


returns_list: List[List[float]] = []
prices_list: List[List[float]] = []
forwards_list: List[float] = []

for ticker in tickers:
    prices = get_prices(ticker, start_date, end_date, 'Close')
    returns = np.log(prices).diff()[1:]
    returns_list.append(returns)
    prices_list.append(prices)
    forwards_list.append(list(prices).pop())

df = pd.DataFrame(returns_list).transpose()
df.columns = tickers
corr_mx = df.corr()

vol_list = []
for price_series in prices_list:
    vol_list.append(get_volatility_estimate(ewma_lambda, price_series))
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
