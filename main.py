import pandas as pd
import yfinance as yf
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sn

tickers = ['MSFT', 'AAPL', 'GOOGL']


def get_prices(ticker, start_date, end_date, column):
    return yf.download(ticker, start=start_date, end=end_date, progress=False)[column]


def something(ewma_lambda, prev, current_price, prev_price):
    return np.sqrt(ewma_lambda * np.square(prev) + (1 - ewma_lambda) * np.square(np.log(current_price / prev_price)))


def get_volatility_estimate(ewma_lambda, prices):
    vol_estimate = 0

    for i in range(0, len(prices) - 1):
        if (i == 0):
            vol_estimate = np.log(prices[1] / prices[0])
        else:
            vol_estimate = something(ewma_lambda, vol_estimate, prices[i + 1], prices[i])

    return vol_estimate


returns_list = []
prices_list = []
forwards_list = []

for ticker in tickers:
    prices = get_prices(ticker, '2020-04-24', '2021-04-24', 'Close')
    returns = np.log(prices).diff()[1:]
    returns_list.append(returns)
    prices_list.append(prices)
    forwards_list.append(list(prices).pop())

df = pd.DataFrame(returns_list).transpose()
df.columns = tickers
corr_mx = df.corr()

vol_list = []
ewma_lambda = .95

for price_series in prices_list:
    vol_list.append(get_volatility_estimate(ewma_lambda, price_series))

vol_mx = pd.DataFrame(np.zeros(corr_mx.shape), columns=tickers)
vol_mx.index = tickers
for i in range(0, len(vol_mx)):
    vol_mx.loc[tickers[i]][tickers[i]] = vol_list[i]

vcv_mx = (vol_mx.dot(corr_mx)).dot(vol_mx)

weight_shape = (1, len(vcv_mx))
position_mx = np.array([50, 100, 10])
forwards_mx = np.array(forwards_list)

notional_value = []
for i in range(0, len(position_mx)):
    notional_value.append(position_mx[i] * forwards_mx[i])

notional_value_mx = np.array(notional_value)
weight_mx = notional_value_mx / np.sum(notional_value_mx)

final = float((weight_mx.dot(vcv_mx.to_numpy())).dot(weight_mx.transpose()))
portfolio_stddev = np.sqrt(final)
shift = portfolio_stddev * np.sqrt(5) * 1.6449

print(np.sum(notional_value_mx) * shift)
