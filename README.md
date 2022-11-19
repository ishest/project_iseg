# About the project

Markowitz is the father of modern portfolio theory. Markowitz formulated the portfolio problem as a choice of the mean and variance of a portfolio of assets. He proved the fundamental theorem of mean variance portfolio theory, namely holding constant variance, maximize expected return, and holding constant expected return minimize variance. These two principles led to the formulation of an efficient frontier from which the investor could choose his or her preferred portfolio, depending on individual risk return preferences. The important message of the theory was that assets could not be selected only on characteristics that were unique to the security. Rather, an investor had to consider how each security co-moved with all other securities. Furthermore, taking these co-movements into account resulted in an ability to construct a portfolio that had the same expected return and less risk than a portfolio constructed by ignoring the interactions between securities. (Elton & Gruber, 1997b)
One of the important contributions of the fourth industrial revolution is the introduction of robo advisors as alternates to conventional mutual funds. Robo advisors are mechanized platforms that use automated algorithms to provide financial advice to investors. (Tao et al., 2021b) 
This study concentrated on building a Robo Advisor based on Modern Portfolio Theory – Mean Variance Theory by Markowitz to see the return diversification when an investor with different risk profiles and preferences uses the built Robo Advisor for different groups of stocks based in different types of markets such as technology, health, consumer, finance and Portuguese. 

# Empirical Work

The main goal of the project is to understand how Robo Advisors output differentiates for different investor profiles defined by parameters of maximum drawdown and risk profile of investor while calculating possible outcomes such as weights of the predefined assets and investor performance between the years 2012 to 2022. 

# Data Collection
All historical data is downloaded with the tool of Yahoo Finance Library created for Python users.
Data Preparation 
Downloaded data is being stored on the users’ computer to being used later for calculations. The data file type is Comma Separated Values (.csv). Yahoo Finance provides daily historical price data starting from 2012 to 2022 without any missing values if the selected stock has been operated in the selected market between these years. 

## Exploring Data
To understand different outcomes for different sectors as mentioned before selected stocks are decided by mainly top volumed stocks of relevant markets with low volatility. 

![image](https://user-images.githubusercontent.com/20598749/202874600-f98136fc-7905-4f44-8714-746b4f7c6bad.png)

<img width="233" alt="image" src="https://user-images.githubusercontent.com/20598749/202874634-a390c4ae-4f2b-4361-9b96-43bffc341dde.png"> <img width="232" alt="image" src="https://user-images.githubusercontent.com/20598749/202874650-a7fbc5d3-12d4-4e39-9607-f246c8f1c5d0.png">

# Data Modelling
Since there is no ideal time period horizon to obtain reliable input data considering average returns, standard deviations and covariances we considered the total time period as 10 years. 
There are many different techniques to deduce Efficient Frontier, analysts sometimes might modify historical estimates so that predictions for the future prices may have better outcomes, other than this, to obtain estimates widely used method is collecting historical data as an input.
In other words, computing Efficient Frontier is sensitive to input data estimates, so we used proposed data which is real historic data without any modifications to see actual outcomes when Portfolio Theory of Markowitz is applied.
Evaluation
The Markowitz theory has the following assumptions (Markowitz, 1952): investors are rational and want to maximize their utility, investors have access to all information needed, markets are efficient, investors are risk-averse and base their decisions accordingly, and for a given level of risk, investors prefer higher returns to lower returns. Some of these assumptions are unrealistic because not all investors have the same investment strategies and not all are risk averse. 
The volatility or risk was derived from the standard deviation of the prices. The correlation between assets is also relevant to construct a portfolio because when assets are less correlated a portfolio is more diversified, leading to higher expected returns and lower risks.
The return of a financial security is the rate computed based on what an investment generates during a certain period of time, where we include the capital gains/losses and the cash-flows it may generate (dividends, in the case of stocks). We can calculate the returns as the difference between an asset price at the end and in the beginning of a selected period, divided by the price of the asset at the beginning of the selected period, 
<br> $R_it=  (P_it-P_(it-1),)/P_it$<br> 
where Rit is the return of asset i on moment t; Pit is the asset i price on moment t; and $Pit − 1$ is the asset i price on moment t-1. 
<br> If t represents a week time interval, weekly returns data are obtained. In addition, the arithmetic mean of the return of the period is calculated as <br> $R_it=(∑_(t=1)^T*(R_it )/T $<br> ,
where T is the number of observations.
<br> 
Markowitz (1952, 1959) introduced the concept of risk and assumed that risk is measured by the variance (or by the standard deviation from the average return), given by 
<br> $σ_i^2=(∑_(t=1)^T*〖〖(R〗_it-R_i)〗^2 )/(T-1)$ <br> 

Covariance enhances the influence of an asset on other assets, with different characteristics, in the determination of the variance of a portfolio. It measures how returns on assets move together. In this context, the correlation is a simple measure to standardize covariance, scaling it with a range of −1 to +1. The covariance concept (Markowitz 1952) puts in evidence the importance of diversification in the choice of the optimal portfolio.
In order to avoid complete liquidation of the portfolio, we implemented maximum withdraw selection of the user over the course of portfolio with maximum value of 70% and created 5 different risk profiles to evaluate 5 different approaches over weightages of stocks and returns.
Deployment The project can be found as an application over the following web-page:
<br> https://robo-iseg.streamlit.app/<br> 
Users must agree with terms and conditions and acknowledge that this project is not built for serving any kind of financial advice and it is only for the information and learning purposes. Then the stock group, maximum drawdown limit and risk profile should be selected to obtain efficient frontier graph, weightage of portfolio, back test of the optimal portfolio and statistics such as Annualized Return, Annualized Volatility, Skewness Index, Kurtosis Index, Cornish Fisher VaR, Historic cVaR, Sharpe Ratio and Maximum Drawdown. These statistics integrated to the project with separate functions. 

