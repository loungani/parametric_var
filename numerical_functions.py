from typing import List

import numpy as np
import pandas as pd


# TODO: Error handling in return calc? (Zero division and negative prices)
def calculate_percentage_return(current_price, next_price):
    return next_price / current_price - 1


def calculate_log_return(current_price, next_price):
    return np.log(next_price / current_price)


def calculate_returns(dataframe, calc_type):
    return_series_list = []
    if calc_type == "percentage":
        func = calculate_percentage_return
    elif calc_type == "log":
        func = calculate_log_return
    else:
        raise ValueError("Bad return calculation specified.")

    for ticker in dataframe.columns:
        price_series = dataframe[ticker]
        return_series = [func(current_price, next_price) for current_price, next_price
                         in zip(price_series[:-1], price_series[1:])]
        return_series_list += [return_series]
    return pd.DataFrame(dict(zip(dataframe.columns, return_series_list)), index=dataframe.index[1:])


# TODO: error handling for undefined log returns
def get_ew_return(ewma_lambda, prev_estimate, current_return) -> float:
    return np.sqrt(ewma_lambda * np.square(prev_estimate) + (1 - ewma_lambda) * np.square(current_return))


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


# TODO: would like to be able to export diagnostics for EWMA returns
def get_volatility_estimate(ewma_lambda, returns_series, averaging_type) -> float:
    if averaging_type == "simple":
        # TODO: Check to see if should be ddof = 1 for stddev calc
        return float(np.std(returns_series))
    elif averaging_type == "ewma":
        vol_estimate = returns_series[0]
        for i in range(1, len(returns_series)):
            vol_estimate = get_ew_return(ewma_lambda, vol_estimate, returns_series[i])
        return vol_estimate
    else:
        raise ValueError("Bad averaging type passed to get_volatility_estimate")


def get_vol_mx(returns, ewma_lambda, averaging_type):
    vol_list = []
    for column in returns:
        vol_list.append(get_volatility_estimate(ewma_lambda, returns[column], averaging_type))
    vol_mx = np.identity(len(vol_list))
    np.fill_diagonal(vol_mx, np.array(vol_list))
    vol_mx = pd.DataFrame(vol_mx, columns=returns.columns, index=returns.columns)
    return vol_mx


def get_corr_mx(df, correlation_type, tickers, ewma_lambda):
    df = df.copy()  # Don't want to modify original
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


def run_principal_components_analysis(num_components: int, eigenvalue_df, eigenvector_df):
    reduced_eigenvalue_array = eigenvalue_df['eigenvalue'].copy()
    reduced_eigenvalue_array[num_components:] = 0
    reduced_eigenvalue_matrix = np.identity(len(eigenvalue_df))
    np.fill_diagonal(reduced_eigenvalue_matrix, np.array(reduced_eigenvalue_array))
    reduced_covariance_matrix = \
        eigenvector_df.dot(reduced_eigenvalue_matrix).dot(np.transpose(eigenvector_df))
    return pd.DataFrame(reduced_eigenvalue_matrix), pd.DataFrame(reduced_covariance_matrix)


def calculate_valuations(prices_df, tickers, positions):
    positions_detail = pd.DataFrame({'ticker': tickers, 'position': positions})
    valuations: List[float] = []
    for date in prices_df.index:
        valuation = 0
        for idx, price in enumerate(prices_df.loc[date]):
            valuation += price * positions_detail.loc[idx]['position']
        valuations += [valuation]
    valuations_df = pd.DataFrame({'Portfolio Valuation': valuations}, index=prices_df.index)
    return valuations_df
