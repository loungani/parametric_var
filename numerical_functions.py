from typing import List

import numpy as np
import pandas as pd


def get_ew_return(ewma_lambda, prev, current_price, prev_price) -> float:
    return np.sqrt(ewma_lambda * np.square(prev) + (1 - ewma_lambda) * np.square(np.log(current_price / prev_price)))


def get_volatility_estimate(ewma_lambda, prices, averaging_type) -> float:
    if averaging_type == "simple":
        return float(np.std(np.log(prices).diff()[1:]))
    elif averaging_type == "ewma":
        vol_estimate = np.log(prices[1] / prices[0])
        for i in range(1, len(prices) - 1):
            vol_estimate = get_ew_return(ewma_lambda, vol_estimate, prices[i + 1], prices[i])
        return vol_estimate
    else:
        raise ValueError("Bad averaging type passed to get_volatility_estimate")


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


def get_corr_mx(df, correlation_type, tickers, ewma_lambda):
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


def get_vol_mx(prices_df, ewma_lambda, averaging_type):
    vol_list = []
    for column in prices_df:
        vol_list.append(get_volatility_estimate(ewma_lambda, prices_df[column], averaging_type))
    vol_mx = np.identity(len(vol_list))
    np.fill_diagonal(vol_mx, np.array(vol_list))
    vol_mx = pd.DataFrame(vol_mx, columns=prices_df.columns, index=prices_df.columns)
    return vol_mx


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
