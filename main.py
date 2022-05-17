from typing import List

import pandas as pd
import yfinance as yf
import numpy as np
import scipy.stats as st

# globals / inputs
ewma_lambda: float = .95
tickers: List[str] = []
positions: List[float] = []
confidence_level: float
holding_period: int
start_date: str
end_date: str


def get_prices(ticker, start_date, end_date, column):
    return yf.download(ticker, start=start_date, end=end_date, progress=False)[column]


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


def create_ewma_weights(length, ewma_lambda, ascending):
    mx = np.ones(length)
    mx[0] = 1 - ewma_lambda
    for i in range(1, len(mx)):
        mx[i] = mx[i - 1] * ewma_lambda

    # TODO: figure out how best to handle ascending vs descending dates throughout script
    if ascending:
        return np.flip(mx)
    else:
        return mx


def get_corr_mx(df, correlation_type):
    if correlation_type == 'simple':
        return df.corr()
    elif correlation_type == 'ewma':
        df['w'] = create_ewma_weights(len(df), ewma_lambda, True)
        for ticker in tickers:
            df[ticker + "^2 * w"] = df[ticker].apply(np.square) * df['w']
            df[ticker + " * w"] = df[ticker] * df['w']
        corr_mx = np.ones((len(tickers), len(tickers)))
        for i in range(0, len(tickers)):
            for j in range(0, len(tickers)):
                numerator = np.sum(df[tickers[i]] * df[tickers[j]] * df['w'])
                denominator_left = np.sqrt(np.sum(df[tickers[i]].apply(np.square) * df['w']))
                denominator_right = np.sqrt(np.sum(df[tickers[j]].apply(np.square) * df['w']))
                denominator = denominator_left * denominator_right
                corr_mx[i][j] = numerator / denominator
        corr_mx = pd.DataFrame(corr_mx)
        corr_mx.index = tickers
        corr_mx.columns = tickers
        return corr_mx
    else:
        raise ValueError("Bad correlation type specified")


def get_arguments():
    io = ''
    while io != 'Y':
        t = input('Add ticker:')
        p = float(input('Add position:'))
        tickers.append(t)
        positions.append(p)
        io = input("Finished? (Y/N")
    start_date = input("Enter start date: (YYYY-MM-DD)")
    end_date = input("Enter end_date: YYYY-MM-DD")
    confidence_level = float(input("Enter confidence level as decimal."))
    holding_period = int(input("Enter holding period as integer."))
    return tickers, positions, start_date, end_date, confidence_level, holding_period


tickers, positions, start_date, end_date, confidence_level, holding_period = get_arguments()

# main body of code

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
corr_mx = get_corr_mx(df, "ewma")

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