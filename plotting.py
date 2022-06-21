import matplotlib.pyplot as plt
import scipy.stats
import numpy as np


def line_plots(line_plot_list: list, series_labels: list, ylabel: str, xlabel: str, title: str):
    plt.figure(figsize=(14, 10), dpi=80)
    for series in line_plot_list:
        plt.plot(series)
    plt.legend(loc='best', labels=series_labels)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


def stock_return_histogram(stock_returns, ticker: str, mu: float, sigma: float, calc_type: str):
    plt.figure(figsize=(14, 10), dpi=80)
    plt.hist(stock_returns, bins=50, density=True)

    # Plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    plt.plot(x, scipy.stats.norm.pdf(x, mu, sigma), 'k')
    plt.title(ticker + ' ' + calc_type + ' returns and fitted distribution ~N('
              + f'{mu:,.3f}' + ', ' + f'{sigma:,.3f}' + ')')
    plt.xlabel(calc_type + ' return')
    plt.ylabel('Observation density')
    plt.show()
