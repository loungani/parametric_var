import numpy as np
import pandas as pd
import scipy.stats as st
import user_input
import query_data
import numerical_functions
import helper_functions
import matplotlib.pyplot as plt

# globals / inputs
ewma_lambda: float = .95
confidence_level: float
holding_period: int
start_date: str
end_date: str
specified_column: str = "Close"

# main body of code
tickers, positions, start_date, end_date, confidence_level, holding_period = user_input.get_arguments()
prices, forwards_list = query_data.get(tickers, start_date, end_date, specified_column)
percentage_returns = numerical_functions.calculate_returns(prices, calc_type="percentage")

use_ewma = \
    user_input.get_boolean("Use EWMA method for estimating parameters? Alternative is equally weighted. (Y/N) ")
if use_ewma:
    ewma_lambda = user_input.get_float("Specify attenuation factor between 0 and 1 for EWMA. "
                                       "(Convention: 0.95)", float_min=0, float_max=1)
    averaging_type = "ewma"
else:
    averaging_type = "simple"

corr_mx = numerical_functions.get_corr_mx(percentage_returns, averaging_type, tickers, ewma_lambda)
vol_mx = numerical_functions.get_vol_mx(percentage_returns, ewma_lambda, averaging_type)
vcv_mx = (vol_mx.dot(corr_mx)).dot(vol_mx)

forwards = np.array(forwards_list)
notional_values = positions * forwards
weights = notional_values / np.sum(notional_values)

# TODO: abstract some of this logic since I'm doing it twice
final = float((weights.dot(vcv_mx)).dot(weights.transpose()))
portfolio_stddev = np.sqrt(final)
z_score = st.norm.ppf(1 - confidence_level)
portfolio_log_return = portfolio_stddev * z_score * np.sqrt(holding_period)

portfolio_mean_return: float = 0
return_averages = [0] * len(tickers)
estimate_mean_return = user_input.get_boolean("Estimate mean portfolio return? Alternative will assume mean of 0. "
                                              "(Y/N) ")

# TODO: Look into EWMA calculation for mean returns?
if estimate_mean_return:
    return_averages = [np.average(percentage_returns[ticker]) for ticker in tickers]
    portfolio_mean_return = np.array(return_averages).dot(weights)
portfolio_log_return += portfolio_mean_return * holding_period

portfolio_percentage_return = np.expm1(portfolio_log_return)
var = abs(np.sum(notional_values)) * portfolio_percentage_return * -1

# Diagnostic: full valuation test / rigorous
valuations: pd.DataFrame = numerical_functions.calculate_valuations(prices, tickers, positions)
valuation_log_returns = numerical_functions.calculate_returns(valuations, calc_type="log")
val_corr_mx = numerical_functions.get_corr_mx(valuation_log_returns, averaging_type, ['Portfolio Valuation'],
                                              ewma_lambda)  # should always be 1 matrix
val_vol_mx = numerical_functions.get_vol_mx(valuation_log_returns, ewma_lambda, averaging_type)
val_vcv_mx = (val_vol_mx.dot(val_corr_mx)).dot(val_vol_mx)

val_notional_values = [valuations.iloc[-1]['Portfolio Valuation']]
val_weights = val_notional_values / np.sum(val_notional_values)
val_final = float((val_weights.dot(val_vcv_mx)).dot(val_weights.transpose()))
val_portfolio_stddev = np.sqrt(val_final)
val_log_return = val_portfolio_stddev * z_score * np.sqrt(holding_period)

val_portfolio_mean_return: float = 0
if estimate_mean_return:
    val_portfolio_mean_return = np.average(valuation_log_returns)
val_log_return += val_portfolio_mean_return * holding_period

val_percentage_return = np.expm1(val_log_return)
val_var = abs(np.sum(val_notional_values)) * val_percentage_return * -1

helper_functions.new_line()
helper_functions.output("Total portfolio value: $" + f'{np.sum(notional_values):,.2f}')
helper_functions.output(f"Portfolio standard deviation: {portfolio_stddev:.2%}")
helper_functions.output(f"Portfolio mean return: {portfolio_mean_return:.2%}")
helper_functions.output(f"Associated log return: {portfolio_log_return:.2%}")
helper_functions.output(f"Associated % return: {portfolio_percentage_return:.2%}")
helper_functions.output(str(holding_period) + "-day" + f"{confidence_level: .2%}"
                        + " VaR: $" + f'{var:,.2f}')
helper_functions.output(f"Portfolio standard deviation (full portfolio/rigorous): {val_portfolio_stddev:.2%}")
helper_functions.output(f"Portfolio mean return (full portfolio/rigorous): {val_portfolio_mean_return:.2%}")
helper_functions.output(f"Associated log return (full portfolio / rigorous): {val_log_return:.2%}")
helper_functions.output(f"Associated % return (full portfolio / rigorous): {val_percentage_return:.2%}")
helper_functions.output(str(holding_period) + "-day" + f"{confidence_level: .2%}"
                        + " VaR (full portfolio / rigorous): $" + f'{val_var:,.2f}')

# TODO: Plotting assumed and historical distribution of returns

# det: should always be between 0 and 1. values close to 0 indicate multicollinearity
det = np.linalg.det(corr_mx)
helper_functions.output("Determinant of correlation matrix: " + str(det))
w, v = np.linalg.eig(vcv_mx)  # note: eigenvectors are returned as columns in the matrix
if any([eigenvalue < 0 for eigenvalue in w]):
    helper_functions.output("Warning: negative eigenvalue found. "
                            "Covariance matrix not positive semi-definite.")
eigenvalue_df = pd.DataFrame(w)
eigenvector_df = pd.DataFrame(v)
eigenvalue_df.rename(columns={0: "eigenvalue"}, inplace=True)
eigenvalue_df.sort_values(by=['eigenvalue'], ascending=False, inplace=True)
eigenvalue_df.reset_index(inplace=True)
eigenvalue_df['proportion_of_variance'] = eigenvalue_df['eigenvalue'] / np.sum(eigenvalue_df['eigenvalue'])
eigenvalue_df['cumulative_proportion_of_variance'] = [np.sum(eigenvalue_df['proportion_of_variance'][0:(i + 1)])
                                                      for i in eigenvalue_df.index]

if user_input.get_boolean("View eigenvector plots? (Y/N) "):
    for (eigenvalue, eigenvector) in zip(w, np.transpose(v)):
        plt.title("Eigenvalue: " + str(eigenvalue))
        plt.ylabel("Unit length eigenvector")
        plt.bar(tickers, eigenvector)
        plt.show()

    cumulative_plot = [0]
    cumulative_plot.extend(list(eigenvalue_df['cumulative_proportion_of_variance']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Proportion of variance explained by principal components')
    ax1.bar(["{:.1e}".format(eigenvalue) for eigenvalue
             in eigenvalue_df['eigenvalue']], eigenvalue_df['proportion_of_variance'])
    ax1.set_xlabel("eigenvalue")
    ax1.set_ylabel("proportion of variance explained")
    ax2.plot(cumulative_plot, marker="^")
    ax2.set_xlabel("# of principal components (sorted)")
    ax2.set_ylabel("Cumulative prop. of variance explained")
    ax2.set_ylim([0, 1])
    plt.show()

reduced_eigenvalue_matrix = None
reduced_covariance_matrix = None
if user_input.get_boolean("Reduce dimensionality using PCA? (Y/N) "):
    max_components = len(eigenvalue_df)
    num_components = user_input.get_int("Enter desired number of components (1 to " +
                                        str(max_components) + ") ", int_min=0, int_max=max_components)
    reduced_eigenvalue_matrix, reduced_covariance_matrix = \
        numerical_functions.run_principal_components_analysis(num_components, eigenvalue_df, eigenvector_df)

    adjusted_final = float((weights.dot(reduced_covariance_matrix)).dot(weights.transpose()))
    adjusted_portfolio_stddev = np.sqrt(adjusted_final)
    adjusted_shift = np.expm1(adjusted_portfolio_stddev * z_score) * np.sqrt(holding_period)
    adjusted_var = abs(np.sum(notional_values) * adjusted_shift)

    helper_functions.output(f"PCA-adjusted portfolio standard deviation: {adjusted_portfolio_stddev:.2%}")
    helper_functions.output(f"PCA-adjusted log return: {adjusted_shift:.2%}")
    helper_functions.output(f"PCA-adjusted percentage return: "
                            f"{adjusted_portfolio_stddev * z_score * np.sqrt(holding_period):.2%}")
    helper_functions.output(str(holding_period) + "-day" + f"{confidence_level: .2%}"
                            + " PCA-adjusted VaR: $" + f'{adjusted_var:,.2f}')

if user_input.get_boolean("Export diagnostics? (Y/N) "):
    positions_detail_df = pd.DataFrame(list(zip(tickers, positions, weights, forwards,
                                                notional_values, return_averages)),
                                       columns=['tickers', 'positions', 'weights', 'forwards',
                                                'notional_values', 'average_return'])
    positions_detail_df.set_index('tickers', inplace=True)
    for dataframe, name in zip([positions_detail_df, prices, percentage_returns, valuations,
                                valuation_log_returns, corr_mx, vol_mx, vcv_mx, eigenvalue_df,
                                eigenvector_df],
                               ['positions.csv', 'prices.csv', 'percentage_returns.csv',
                                'valuation_log_returns.csv', 'valuations.csv', 'valuation_returns.csv',
                                'corr_mx.csv', 'vol_mx.csv', 'vcv_mx.csv',
                                'eigenvalues.csv', 'eigenvectors.csv']):
        dataframe.to_csv(name)
    if (reduced_eigenvalue_matrix is not None) and (reduced_covariance_matrix is not None):
        reduced_eigenvalue_matrix.to_csv("reduced_eigenvalue_matrix.csv")
        reduced_covariance_matrix.to_csv("reduced_covariance_matrix.csv")
