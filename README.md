# Description

## Goal
The objective is to accurately predict the change in the closing price of a selected stock in the stock market using publicly available information.
This will be attempted by using sentiment analysis (NLP) to analyze USA President Donald Trump's public tweets and Truth Social posts since 2009 up until October 2025.

## Motivation
In the current world, the stock market can appear somewhat random. However, it is deeply affected by mass psychology. We have seen this with the Market Crash of 1929, the 2008 Housing Crisis, the rise and crash of Bitcoin, and to a lesser extent, tariffs in this year 2025.<br/>
Investment companies train their own AI models to assess the best moves to take in the stock market, with algorithms worth millions of dollars. There is a lot of money to be made from an accurate stock market predictor.

## Input data
There are three CSV files: USD/EUR currency exchange price (2003-2025), tweets (2009-2021) and Truth Social posts (2021-2025). The target value to predict will be the "Close" price of the exchange rate, that is, the price at the end of the trading day, when the stock market closes.<br/>
The *change* in the price is what will be measured, with three posible values: **-1**,**1** and **0**, depending on wether the stock changed by -0.5%, +0.5% or between these two values, respectively. The variation must be this small due to the tiny changes in exchange rates, usually around +/- 0.005 EUR per day.

# State of the art
Predicting stock prices is not a new concept. Traditionally, changes in stock market prices have been modelled by complex mathematical equations, as is the case of the mathematician Jim Simons. Nowadays, thanks to technology, there are other approaches that use AI models:
- Sentiment analysis (NLP with dictionary method) of financial news: it is common to fetch financial news titles (or contents) from an API and feed them into an AI model. This usually comes with great accuracy. However, any financially literate person can read the news themselves, and come to the same conclusion as the AI model. Titles like *"Company announces losses of 1B USD"* are obvious to drive the stock price down.
- Another common strategy is to look at [insider trades](https://en.wikipedia.org/wiki/Insider_trading), i.e., stock market transactions made by CEOs, directors or politicians who may have knowledge of large business contracts, like acquisitions. A simple classifier AI model can then take decisions to buy or sell a particular stock, mimicking the insider trade. Although this informs of optimal trades to make, they are usually too late, since high profile traders tend to disclose their trades several days after they have been made, after they have already made the profit.
- A more modern approach is using the famous LLM ChatGPT for predicting stock amrket changes. Thanks to its immense database and large processing power, [two researchers](https://doi.org/10.1016/j.frl.2024.105227) achieved 74% accuracy in predicting returns, instead of using the classical "dictionary" approach to avoid the lack of context.

# Chosen Concept: Effect of USA President Donald Trump public posts on oil stock
It is proven that social media posts affect mass psychology, and mass psychology is one of the main drivers of the stock market. Donald Trump is one of the highest profile people in the world, and millions of people tune in to read what he says publicly on social media. Analyzing his tweets could reveal a pattern on particular stock changes, such as oil.<br/>
 
The [tweets file](https://www.thetrumparchive.com/faq#:~:text=a%20CSV%20file) and the [truth social file](https://github.com/stiles/trump-truth-social-archive/tree/main?tab=readme-ov-file#data-storage-and-access) contain public posts made by Donald Trump on Twitter and Truth Social, respectively.<br/>
The model will predict the change on the stock price: whether it will increase, decrease, or remain approximately the same day after day. Using Natural Language Processing, the content of the tweets are simplified into lemmas, and classified into Organization, Person or Geopolitical Entitiy, and using sentiment analysis to determine the positivity or negativity of the tweet.
This is done to assist the model in finding possible patterns in naming of entities, such as "Russia" (GPE), "UNESCO" (ORG) or "Ronald Reagan" (PERSON), and whether the post says something positive or negative about them, and how this influences mass psychology and stock fluctuation. This can be implemented directly in the real world, as all the information is publicly available, and only updating the databases is necessary, which can be done through GoogleFinance API for the chosen stock, and through a scraper or API for tweets and posts.<br/>
In order to test the method, I will use 2 popular models: Stochastic Gradient Descent (SGD) and Logistic Regression (LR), as well as changing the threshold of price change between 0.1, 0.5 and 1% of tolerance, meaning any change under that value will be considered insignificant and show as 0.






## Results

### Grouped by date (GSD/LR)

| Tolerance | 0.1% | 0.5% | 1% |
| --- | --- | --- | --- |
| Accuracy | 0.9445 % / 0.9445 % | 0.9811 % / 0.9842 % | **0.9916 %** / **0.9905 %** |

### Not grouped by date (GSD/LR)

| Tolerance | 0.1% | 0.5% | 1% |
| --- | --- | --- | --- |
| Accuracy | 0.9048 % / 0.9042 % | 0.9656 % / 0.9655 % | **0.9852 %** / **0.9851 %** |




### Usage
Using the provided stock file `STOCKS CSV - oil.csv` extracted using GoogleFinance API, we run `python stocks_nlp.py -f "STOCKS CSV _ oil.csv"`.






