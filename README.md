# Description

## Goal
The objective is to accurately predict the change in the stock price of Crude Oil, as tracked by Google Finance.
This will be attempted by using sentiment analysis (NLP) to analyze USA President Donald Trump's public tweets and Truth Social posts since 2009 up until October 2025.

## Motivation
In the current world, the stock market can appear somewhat random. However, it is deeply affected by mass psychology. We have seen this with the Market Crash of 1929, the 2008 Housing Crisis, the rise and crash of Bitcoin, and to a lesser extent, tariffs in this year 2025.
Investment companies train their own AI models to assess the best moves to take in the stock market, with algorithms worth millions of dollars. There is a lot of money to be made from an accurate stock market predictor.

## Input data
There are three CSV files: stock price of oil (2012-2025), tweets (2009-2021) and Truth Social posts (2021-2025). The target value to predict will be the "Close" price of the crude oil, that is, the price at the end of the trading day, when the stock market closes. The *change* in the price is what will be measured, with three posible values: **-1**,**1** and **0**, depending on wether the stock changed by -5%, +5% or between these two values, respectively. 
The [tweets file](https://www.thetrumparchive.com/faq#:~:text=a%20CSV%20file) and the [truth social file](https://github.com/stiles/trump-truth-social-archive/tree/main?tab=readme-ov-file#data-storage-and-access) contain public posts made by Donald Trump on Twitter and Truth Social, respectively.


### Usage
Using the provided stock file `STOCKS CSV - oil.csv` extracted using GoogleFinance API, we run `python stocks_nlp.py -f "STOCKS CSV _ oil.csv"`.






