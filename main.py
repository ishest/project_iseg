import functions as fc
import numpy as np
import pandas as pd
import streamlit as st
from data_load import update
import matplotlib.pyplot as plt
plt.style.use("dark_background")

# plt.style.use('fivethirtyeight')


industries =['Tech', 'Health', 'Consumers', 'Portuguese', 'Finance']

st.title('Robo Advisor based on Modern Portfolio Theory')
st.subheader('created by Igor, Maxwell, Tansu and Rafael')

expander_bar = st.expander("About the project")
expander_bar.markdown("""
* **(c) ISEG, 2022:**
* **Stocks:**  SPBEX,  NYSE, ETF, Currencies
* **Data source:** Source: https://finance.yahoo.com/
* **Investment strategy:** Modern Portfolio Theory and optimization process*
""")

st.image('robo_img.jpeg')
st.text('The information on the website is not financial advice. It is only \n'
        'for learning and educational purposes. \n'
        'Please, agree with these terms and conditions.')


if st.checkbox('I agree'):
        st.write('Welcome to our service')
else:
        st.write('Please, confirm that you are agree with all terms and conditions')

selected_sector = st.selectbox(label="Choose your stocks", options=industries)

if st.button('update data'):
        update()


Max_DD = st.slider('Maximum Drawdown', 0.1, 0.7, 0.4)
Risk_level = st.slider('Risk Profile', 1, 5, 1)

if st.button('Build a portfolio'):

        st.header(f'Efficient Frontier of the {selected_sector} companies')
        data = pd.read_csv('data/'+selected_sector+'.csv', index_col=0)
        data_pct = data.pct_change()

        er = fc.annual_returns(data_pct, 365)
        cov = data_pct.cov()

        # test
        weights = fc.optimal_weights(30, er, cov)

        # Efficient Frontier
        rets = [fc.port_return(w, er) for w in weights]
        vols = [fc.port_volatility(w, cov) for w in weights]
        ef = pd.DataFrame({
                "Returns": rets,
                "Volatility": vols
        })

        fig, ax = plt.subplots()
        for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
                plt.rcParams[param] = '0.9'  # very light grey
        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
                plt.rcParams[param] = '#212946'  # bluish dark grey
        colors = [
                '#08F7FE',  # teal/cyan
                '#FE53BB',  # pink
                '#F5D300',  # yellow
                '#00ff41',  # matrix green
        ]

        ax.plot(ef["Volatility"], ef["Returns"], '-', linewidth=0.7, markersize=12)
        n = er.shape[0]
        w_ew = np.repeat(1 / n, n)
        # print('Weights in EW', w_ew)
        r_ew = fc.port_return(w_ew, er)
        vol_ew = fc.port_volatility(w_ew, cov)
        # display EW
        ax.plot([vol_ew], [r_ew], color='yellow', marker='o', markersize=10, label='Equally Weighted')
        # 'goldenrod'
        w_gmv = fc.gmv(cov)
        # print('Weights in GMV', list(w_gmv))
        r_gmv = fc.port_return(w_gmv, er)
        vol_gmv = fc.port_volatility(w_gmv, cov)
        # display GMV
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10, label='Global Minimum Vol')
        # 'midnightblue'
        ax.set_xlim(left=0)
        w_msr = fc.msr(0, er, cov)
        # print('Weights in MSR', w_msr)
        r_msr = fc.port_return(w_msr, er)
        vol_msr = fc.port_volatility(w_msr, cov)
        # Add CML Capital Market Line
        cml_x = [0, vol_msr]
        cml_y = [0, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', markersize=12, linewidth=2,
                # 'green'
                label='Tangent of Optimal Portfolio')
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Return')

        ax.grid(color='#2A3459')

        ax.legend()

        st.pyplot(fig)
        # finish with EF

        # table of weights

        st.header(f'Percent of the invested capital that should be allocated into each asset')

        weights = list(fc.gmv(cov))
        gmv_portfolio = np.multiply(data_pct, weights).sum(axis=1)

        # weights = list(fc.msr(0, er, cov))

        tickers = pd.DataFrame(data.columns, columns=['Ticker'])
        tickers['weights'] = list(weights)
        st.table(tickers.style.format({'weights': '{:.2%}'}))


        @st.cache
        def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv().encode('utf-8')


        csv = convert_df(tickers)

        st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='my_optimal_portfolio.csv',
                mime='text/csv',
        )



        cashrate = 0
        monthly_cashreturn = (1+cashrate)**(1/12) - 1

        rets_cash = pd.DataFrame(data=monthly_cashreturn, index=gmv_portfolio.index, columns=[0]) # 1 column dataframe
        rets_maxdd = fc.backtest_two_assets(pd.DataFrame(gmv_portfolio), rets_cash, allocator=fc.drawdown_allocator, maxdd=.25, m=3)
        dd_25 = fc.drawdown(rets_maxdd[0])

        # fc.summary_stats(pd.DataFrame(rets_maxdd))

        st.header(f'Backtest of the Optimal Portfolio of the {selected_sector} companies')
        fc.port(Max_DD, Risk_level, gmv_portfolio)

