from typing import List
import helper_functions

def get_arguments():
    tickers: List[str] = []
    positions: List[float] = []

    io = ''
    while io != 'Y':
        try:
            helper_functions.output("Current ticker list: " + " ".join(tickers))
            t = input('Add ticker: ')
            p = float(input('Add position: '))
            tickers.append(t)
            positions.append(p)
            io = input("Finished? (Y/N) ")
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
                raise ValueError("Expected confidence_level strictly between 0 and 1. Received: " + str(confidence_level))
            holding_period = int(input("Enter holding period as integer: "))
            if holding_period < 1:
                raise ValueError("Expected holding period geq 1. Received: " + str(holding_period))
            need_inputs = False
        except Exception as e:
            helper_functions.output("Exception caught in get_arguments() function: ")
            helper_functions.output("Stack trace: " + str(e))
            helper_functions.new_line()

    return tickers, positions, start_date, end_date, confidence_level, holding_period
