import yfinance as yf
import helper_functions
import sys


def get_prices(ticker, start_date, end_date, column):
    try:
        return yf.download(ticker, start=start_date, end=end_date, progress=False)[column]
    except Exception as e:
        helper_functions.output("Error when calling get_prices function.")
        helper_functions.output("ticker: " + ticker)
        helper_functions.output("start_date: " + start_date)
        helper_functions.output("end_date: " + end_date)
        helper_functions.output("column: " + column)
        helper_functions.output("Stack trace: " + str(e))
        sys.exit()