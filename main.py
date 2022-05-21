import numpy as np
import pandas as pd
import scipy.stats as st
import user_input
import query_data
import numerical_functions

# globals / inputs
ewma_lambda: float = .95
confidence_level: float
holding_period: int
start_date: str
end_date: str
specified_column: str = "Close"


# main body of code

tickers, positions, start_date, end_date, confidence_level, holding_period = user_input.get_arguments()
prices_df, returns_df, forwards_list = query_data.get(tickers, start_date, end_date, specified_column)
corr_mx = numerical_functions.get_corr_mx(returns_df, "ewma", tickers, ewma_lambda)
vol_mx = numerical_functions.get_vol_mx(prices_df, ewma_lambda, "ewma")
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