import pandas as pd
import yfinance as yf
import helper_functions
import sys
from typing import List
import numpy as np


def get_prices(ticker, start_date, end_date, specified_column):
    try:
        return yf.download(ticker, start=start_date, end=end_date, progress=False)[specified_column]
    except Exception as e:
        helper_functions.output("Error when calling get_prices function.")
        helper_functions.output("ticker: " + ticker)
        helper_functions.output("start_date: " + start_date)
        helper_functions.output("end_date: " + end_date)
        helper_functions.output("specified_column: " + specified_column)
        helper_functions.output("Stack trace: " + str(e))
        sys.exit()


def get(tickers, start_date, end_date, specified_column):
    forwards_list: List[float] = []
    prices_df = pd.DataFrame()

    first = True
    for ticker in tickers:
        prices = get_prices(ticker, start_date, end_date, specified_column)

        if first:
            prices_df = prices.to_frame().rename(columns={specified_column: ticker})
            first = False
        else:
            prices_df = prices_df.join(prices.to_frame().rename(columns={specified_column: ticker}))

        forwards_list.append(list(prices).pop())
    return prices_df, forwards_list
