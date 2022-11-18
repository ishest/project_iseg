import functions as fc
import numpy as np
import pandas as pd
import streamlit as st
from data_load import update

industries =['Tech', 'Health', 'Consumer Industry', 'Portugal', 'Finance']

st.title('Robo Advisor based on Modern Portfolio Theory')
st.subheader('created by Maaaaxwell, RRRRRafael, TAAAAnsu, and Igooooor')
st.subheader('(c) ISEG, 2022')
expander_bar = st.expander("About the project")
expander_bar.markdown("""
* **Stocks:**  SPBEX,  NYSE, ETF, Currencies
* **Data source:** Source: Yahoo Finance, https://exchange.iex.io/.
* **Investment strategy:** Modern Portfolio Theory and optimization process*
""")

st.image('robo_img.png')
st.text('The information on the website is not a financial advice. It is only \n'
        'for the information and learning purposes. \n'
        'Please, agree with these terms and conditions')


if st.checkbox('I agree'):
        st.write('Welcome to our service')
else:
        st.write('Please, confirm that you are agree')

selected_sector = st.selectbox(label="Choose your stocks", options=industries)

if st.button('update data'):
        update()


Max_DD = st.slider('Maximum Drawdown', 0.1, 0.7, 0.4)
Risk_level = st.slider('Risk Profile', 1, 5, 1)

if st.button('Build a portfolio'):

        st.header(f'Investment Portfolio on {selected_sector} companies')



        data = pd.read_csv(selected_sector+'.csv', index_col=0)
        data_pct = data.pct_change()

        er = fc.annualize_rets(data_pct, 365)
        cov = data_pct.cov()

        weights = list(fc.gmv(cov))
        gmv_portfolio = np.multiply(data_pct, weights).sum(axis=1)


        cashrate = 0
        monthly_cashreturn = (1+cashrate)**(1/12) - 1


        rets_cash = pd.DataFrame(data=monthly_cashreturn, index=gmv_portfolio.index, columns=[0]) # 1 column dataframe
        rets_maxdd25 = fc.bt_mix(pd.DataFrame(gmv_portfolio), rets_cash, allocator=fc.drawdown_allocator, maxdd=.25, m=3)
        dd_25 = fc.drawdown(rets_maxdd25[0])

        # ax = dd_25["Wealth"].plot(figsize=(10, 5), label="MaxDD 25%", color="cornflowerblue", legend=True, linewidth=1)
        # dd_25["Peaks"].plot(ax=ax, color="cornflowerblue", ls=":", linewidth=1)

        fc.summary_stats(pd.DataFrame(rets_maxdd25))


        fc.port(Max_DD, Risk_level, gmv_portfolio)



        st.subheader('Portfolio statistics')
