from typing import List


def get_arguments():
    tickers: List[str] = []
    positions: List[float] = []

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
