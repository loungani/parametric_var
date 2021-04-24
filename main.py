from typing import List, Any

import pandas as pd
import yfinance as yf
import numpy as np
import scipy.stats as st

# import matplotlib.pyplot as plt
# import seaborn as sn

# globals / inputs
tickers: List[str] = ['MSFT', 'AAPL', 'GOOGL']
ewma_lambda: float = .95
confidence_level: float = .95
holding_period: int = 5
start_date: str = '2020-04-24'
end_date: str = '2021-04-24'
position_mx = np.array([50, 100, 10])


def get_prices(ticker, start_date, end_date, column):
    return yf.download(ticker, start=start_date, end=end_date, progress=False)[column]


def get_ew_return(ewma_lambda, prev, current_price, prev_price):
    return np.sqrt(ewma_lambda * np.square(prev) + (1 - ewma_lambda) * np.square(np.log(current_price / prev_price)))


def get_volatility_estimate(ewma_lambda, prices):
    vol_estimate = np.log(prices[1] / prices[0])
    for i in range(1, len(prices) - 1):
        vol_estimate = get_ew_return(ewma_lambda, vol_estimate, prices[i + 1], prices[i])
    return vol_estimate


returns_list: List[float] = []
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

weight_shape = (1, len(vcv_mx))
forwards_mx = np.array(forwards_list)

notional_value = []
for position, forward in zip(position_mx, forwards_mx):
    notional_value.append(position * forward)

notional_value_mx = np.array(notional_value)
weight_mx = notional_value_mx / np.sum(notional_value_mx)

final = float((weight_mx.dot(vcv_mx)).dot(weight_mx.transpose()))
portfolio_stddev = np.sqrt(final)
z_score = st.norm.ppf(confidence_level)
shift = portfolio_stddev * np.sqrt(holding_period) * z_score

var = np.sum(notional_value_mx) * shift
print("$" + f'{var:,.2f}')
