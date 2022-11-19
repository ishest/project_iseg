import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# import scipy
import math
import time
from scipy.stats import norm
from scipy.optimize import minimize
# import statsmodels.api as sm
import streamlit as st
import yfinance as yf

def annualize_rets(r, periods_per_year):

    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    return r.std()*(periods_per_year**0.5)


def portfolio_return(weights, returns):
    return weights.T.dot(returns)


def portfolio_vol(weights, covmat):
    return np.dot(weights.T.dot(covmat), weights) ** 0.5


def minimize_vol(target_return, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    return_is_target = {  # return that we expect and it is constrain to our function
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }

    # weights can't exceed 100%

    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    results = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds
                       )
    return results.x


def optimal_weights(n_points, er, cov):
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights


# Max Sharpe Ratio
def msr(riskfree_rate, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n  # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
                        }

    def neg_sharpe(weights, riskfree_rate, er, cov):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    results = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return results.x


# Global minimum Volatility
def gmv(cov):
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


# Descriptive formulas

def drawdown(return_series: pd.Series):
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    # convert the annual riskfree rate to per period
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def skewness(r):
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 3).mean()
    return exp / sigma_r ** 3


def kurtosis(r):
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r ** 4).mean()
    return exp / sigma_r ** 4


def var_historic(r, level=5):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def var_gaussian(r, level=5, modified=False):
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level / 100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z ** 2 - 1) * s / 6 +
             (z ** 3 - 3 * z) * (k - 3) / 24 -
             (2 * z ** 3 - 5 * z) * (s ** 2) / 36
             )
    return -(r.mean() + z * r.std(ddof=0))


def cvar_historic(r, level=5):
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")


def summary_stats(r, riskfree_rate=0.0):
    ann_r = r.aggregate(annualize_rets, periods_per_year=252)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=252)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=252)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })


def drawdown_allocator(risky_part, free_risk, maxdd, m=3):

    n_steps, n_scenarios = risky_part.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=risky_part.index, columns=risky_part.columns)
    for step in range(n_steps):
        floor_value = (1-maxdd)*peak_value ### Floor is based on Prev Peak
        cushion = (account_value - floor_value) / account_value
        risky_part_w = (m * cushion).clip(0, 1) # same as applying min and max
        free_risk_w = 1 - risky_part_w
        risky_part_alloc = account_value * risky_part_w
        free_risk_alloc = account_value * free_risk_w
        # recompute the new account value and prev peak at the end of this step
        account_value = risky_part_alloc*(1 + risky_part.iloc[step]) + free_risk_alloc * (1 + free_risk.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = risky_part_w
    return w_history


def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 should have the same shape")
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights with a different shape than the returns")
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix


def port(Max_DD, Risk_level, gmv_portfolio):
    def drawdown_allocator(psp_r, ghp_r, maxdd=Max_DD, m=Risk_level):
        n_steps, n_scenarios = psp_r.shape
        account_value = np.repeat(1, n_scenarios)
        floor_value = np.repeat(1, n_scenarios)
        peak_value = np.repeat(1, n_scenarios)
        w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
        for step in range(n_steps):
            floor_value = (1 - maxdd) * peak_value  ### Floor is based on Prev Peak
            cushion = (account_value - floor_value) / account_value
            psp_w = (m * cushion).clip(0, 1)  # same as applying min and max
            ghp_w = 1 - psp_w
            psp_alloc = account_value * psp_w
            ghp_alloc = account_value * ghp_w
            # recompute the new account value and prev peak at the end of this step
            account_value = psp_alloc * (1 + psp_r.iloc[step]) + ghp_alloc * (1 + ghp_r.iloc[step])
            peak_value = np.maximum(peak_value, account_value)
            w_history.iloc[step] = psp_w
        return w_history

    cashrate = 0
    monthly_cashreturn = (1 + cashrate) ** (1 / 12) - 1
    rets_cash = pd.DataFrame(data=monthly_cashreturn, index=gmv_portfolio.index, columns=[0])  # 1 column dataframe

    rets_maxdd25 = bt_mix(pd.DataFrame(gmv_portfolio), rets_cash, allocator=drawdown_allocator, maxdd=Max_DD,
                          m=Risk_level)
    dd_25 = drawdown(rets_maxdd25[0])

    fig, ax = plt.subplots()

    ax = dd_25["Wealth"].plot(figsize=(10, 5), title='Investor Performance',
                              label='MaxDD: {}%\nRisk Profile: {}'.format(round(Max_DD * 100, 0), Risk_level),
                              color="cornflowerblue", legend=True, linewidth=1)


    dd_25["Peaks"].plot(ax=ax, color="red", ls=":", linewidth=1)

    st.pyplot(fig)
    # st.pyplot(dd_25)

    stats = pd.DataFrame(summary_stats(rets_maxdd25)).style.hide_index()
    #     print('\n', pd.DataFrame(pkt.summary_stats(rets_maxdd25)).T)

    st.subheader('Portfolio statistics')

    hide_dataframe_row_index = """
                <style>
                .row_heading.level0 {display:none}
                .blank {display:none}
                </style>
                """

    # Inject CSS with Markdown
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

    st.table(stats.format({'Annualized Return': '{:.2%}',
                           'Annualized Vol': '{:.2%}',
                           'Cornish-Fisher VaR (5%)':'{:.2%}',
                           'Historic CVaR (5%)':'{:.2%}',
                           'Max Drawdown':'{:.2%}'}))




