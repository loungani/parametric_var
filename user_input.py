from typing import List
import helper_functions
import pandas as pd


def get_boolean(message):
    need_input = True
    while need_input:
        io = input(message)
        if io == 'Y':
            return True
        elif io == 'N':
            return False
        else:
            helper_functions.output("Expected Y or N. Received: " + io)


def get_arguments():
    tickers: List[str] = []
    positions: List[float] = []

    need_inputs = True
    while need_inputs:
        try:
            helper_functions.output("Current ticker list: " + " ".join(tickers))
            t = input('Add ticker: ')
            p = float(input('Add position: '))
            if t in tickers:
                raise ValueError("ticker: " + str(t) + " already found in tickers list.")
            tickers.append(t)
            positions.append(p)
            need_inputs = get_boolean("Enter another ticker? (Y/N) ")
            helper_functions.new_line()
        except Exception as e:
            helper_functions.output("Ticker/position not accepted.")
            helper_functions.output("Stack trace: " + str(e))

    need_inputs = True
    while need_inputs:
        try:
            start_date = input("Enter start date: (YYYY-MM-DD) ")
            end_date = input("Enter end date: (YYYY-MM-DD) ")
            confidence_level = float(input("Enter confidence level as decimal: "))
            if confidence_level >= 1 or confidence_level <= 0:
                raise ValueError(
                    "Expected confidence_level strictly between 0 and 1. Received: " + str(confidence_level))
            holding_period = int(input("Enter holding period as integer: "))
            if holding_period < 1:
                raise ValueError("Expected holding period geq 1. Received: " + str(holding_period))
            need_inputs = False
        except Exception as e:
            helper_functions.output("Exception caught in get_arguments() function: ")
            helper_functions.output("Stack trace: " + str(e))
            helper_functions.new_line()

    return tickers, positions, start_date, end_date, confidence_level, holding_period


def export_diagnostics(positions_detail_df, prices_df, returns_df, corr_mx, vol_mx,
                       vcv_mx, eigenvalue_df, eigenvector_df):
    positions_detail_df.to_csv("positions_detail.csv")
    prices_df.to_csv("prices.csv")
    returns_df.to_csv("returns.csv")
    corr_mx.to_csv("correlation_matrix.csv")
    vol_mx.to_csv("volatility_matrix.csv")
    vcv_mx.to_csv("variance_covariance_matrix.csv")
    eigenvalue_df.to_csv("eigenvalues.csv")
    eigenvector_df.to_csv("eigenvectors.csv")
