import numpy as np
import pandas as pd


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
