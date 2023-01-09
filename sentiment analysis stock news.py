from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

newsURL = "https://finviz.com/quote.ashx?t="

'''numEntries = int(input("Enter number of companies you want to analyze (upto 4):"))
for ii in range(numEntries):
    company = input("Enter the company ticker :")
    tickers.append(company)'''

tickers = ['AMZN','GOOGL']

news_tables = {}
for ticker in tickers:
    completeURL = newsURL + ticker
    
    req = Request(url=completeURL, headers={'user-agent' : 'my-app'})
    response = urlopen(req)
    
    html = BeautifulSoup(response, features = "lxml")
    
    #now we need to navigate to the table with the news
    #we will fill our dictionary with the conents of this table
    news_table = html.find(id = 'news-table')
    news_tables[ticker] = news_table

data = []

for ticker, news_table in news_tables.items():
    
    for row in news_table.findAll('tr'):
        news_title = row.a.get_text()
        date_stamps = row.td.text.split(' ')
        
        if(len(date_stamps) == 1):
            time = date_stamps[0]
        else:
            date = date_stamps[0]
            time = date_stamps[1]
            
        data.append([ticker, date, time, news_title])
        
dataframe = pd.DataFrame(data, columns = ['ticker', 'date', 'time', 'news'])

vader = SentimentIntensityAnalyzer()

score_calc = lambda news: vader.polarity_scores(news)['compound']

dataframe['compound'] = dataframe['news'].apply(score_calc)

dataframe['date'] = pd.to_datetime(dataframe.date).dt.date

plt.figure(figsize=(10,8))

plot_df = dataframe.groupby(['ticker','date']).mean()
plot_df = plot_df.unstack()
plot_df = plot_df.xs('compound', axis = "columns").transpose()

plot_df.plot(kind='bar')
plt.show()