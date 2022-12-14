import pandas as pd
import yfinance as yf
import time


tech_list = ['AAPL', 'ADBE', 'AMD', 'NVDA', 'IBM', 'CSCO', 'CTSH', 'CTXS', 'HPQ']

health_list = ['UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'NVS', 'DHR','BIIB']

consumer_list = ['WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'UL','DIS']

portugal_list = ['EDPR.LS', 'NOS.LS', 'JMT.LS', 'SON.LS', 'ALTR.LS', 'NVG.LS', 'SEM.LS', 'GALP.LS','SLBEN.LS','COR.LS']

financial_list = ['V', 'JPM', 'C', 'MA', 'GS', 'SCHW', 'BAC', 'MS', 'WFC']

list_industries = [tech_list, health_list, consumer_list, portugal_list, financial_list]
name_inds = ['Tech', 'Health', 'Consumers', 'Portuguese', 'Finance']

start_date = time.strftime("2012-11-01")


def update():
    for i in range(len(name_inds)):
        data = pd.DataFrame(columns=list_industries[i])
        data = yf.download(list_industries[i], start=start_date)['Adj Close']
        data.to_csv('data/'+str(name_inds[i])+'.csv')

