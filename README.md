# Description

## Goal
The objective is to accurately predict the change in the closing price of a selected stock in the stock market using publicly available information.
This will be attempted by using sentiment analysis (NLP) to analyze USA President Donald Trump's public tweets and Truth Social posts since 2009 up until October 2025.

## Motivation
In the current world, the stock market can appear somewhat random. However, it is deeply affected by mass psychology. We have seen this with the Market Crash of 1929, the 2008 Housing Crisis, the rise and crash of Bitcoin, and to a lesser extent, tariffs in this year 2025.<br/>
Investment companies train their own AI models to assess the best moves to take in the stock market, with algorithms worth millions of dollars. There is a lot of money to be made from an accurate stock market predictor.

## Input data
There are three CSV files: USD/EUR currency exchange price (2003-2025), tweets (2009-2021) and Truth Social posts (2021-2025). The target value to predict will be the "Close" price of the exchange rate, that is, the price at the end of the trading day, when the stock market closes.<br/>
The *change* in the price is what will be measured, with three posible values: **-1**,**1** and **0**, depending on wether the stock changed by -0.01%, +0.01% or between these two values, respectively. The variation must be this small due to the tiny changes in exchange rates, usually around +/- 0.005 EUR per day.

# State of the art
Predicting stock prices is not a new concept. Traditionally, changes in stock market prices have been modelled by complex mathematical equations, as is the case of the mathematician Jim Simons. Nowadays, thanks to technology, there are other approaches that use AI models:
- Sentiment analysis (NLP with dictionary method) of financial news: it is common to fetch financial news titles (or contents) from an API and feed them into an AI model. This usually comes with great accuracy. However, any financially literate person can read the news themselves, and come to the same conclusion as the AI model. Titles like *"Company announces losses of 1B USD"* are obvious to drive the stock price down.
- Another common strategy is to look at [insider trades](https://en.wikipedia.org/wiki/Insider_trading), i.e., stock market transactions made by CEOs, directors or politicians who may have knowledge of large business contracts, like acquisitions. A simple classifier AI model can then take decisions to buy or sell a particular stock, mimicking the insider trade. Although this informs of optimal trades to make, they are usually too late, since high profile traders tend to disclose their trades several days after they have been made, after they have already made the profit.
- A more modern approach is using the famous LLM ChatGPT for predicting stock amrket changes. Thanks to its immense database and large processing power, [two researchers](https://doi.org/10.1016/j.frl.2024.105227) achieved 74% accuracy in predicting returns, instead of using the classical "dictionary" approach to avoid the lack of context.

# Chosen Concept: Effect of USA President Donald Trump public posts on oil stock
It is proven that social media posts affect mass psychology, and mass psychology is one of the main drivers of the stock market. Donald Trump is one of the highest profile people in the world, and millions of people tune in to read what he says publicly on social media. Analyzing his tweets could reveal a pattern on particular stock changes, such as oil.<br/>
 
The [tweets file](https://www.thetrumparchive.com/faq#:~:text=a%20CSV%20file) and the [truth social file](https://github.com/stiles/trump-truth-social-archive/tree/main?tab=readme-ov-file#data-storage-and-access) contain public posts made by Donald Trump on Twitter and Truth Social, respectively.<br/>
The model will predict the change on the stock price: whether it will increase, decrease, or remain approximately the same day after day. Using Natural Language Processing, the content of the tweets are simplified into lemmas, and classified into Organization, Person or Geopolitical Entitiy, and using sentiment analysis to determine the positivity or negativity of the tweet.<br/>
This is done to assist the model in finding possible patterns in naming of entities, such as "Russia" (GPE), "UNESCO" (ORG) or "Ronald Reagan" (PERSON), and whether the post says something positive or negative about them, and how this influences mass psychology and stock fluctuation. This can be implemented directly in the real world, as all the information is publicly available, and only updating the databases is necessary, which can be done through GoogleFinance API for the chosen stock, and through a scraper or API for tweets and posts.<br/>
In order to test the method, I will use 2 popular models: Stochastic Gradient Descent (SGD) and Logistic Regression (LR).<br/>
The accuracy of the model will be tested by using the `accuracy_score` function, and a custom function `check_accuracy`, which will check the number of correct guesses when the target value is not zero. This allows to focus only on actual changes in price, and not just remaining constant. If the value matches (-1==-1 or 1==1) then the number increases by 1, and if the predicted value was 0 but the correct answer was 1, it will increase by 0.5, since it is an acceptable outcome (the model expects the stock to remain constant but it increases, which is still good news if we are currently holding that stock).<br/>

```
##Measure accuracy for predicting changes different from 0
def check_accuracy(goal:numpy.ndarray,prediction:numpy.ndarray):
    acc=0.0
    for g,p in zip(goal,prediction):
        if g!=0:
            if g==p:
                acc+=1
            elif g==1 and p==0:
                acc+=0.5

    return float(acc/goal.shape[0])
```

## Results
Both the `accuracy_score` and `check_accuracy` functions calculate a similar accuracy of around 44%, and in every occasion the accuracy is higher for the dataframe with grouped tweets by dates. LR shows lower accuracy in every case.<br/>
Using a function to calculate the actual ammount of money won or lost, `test_profit`, we can see the actual impact and usefulness of the model. 

<img width="639" height="479" alt="Predicted choices" src="https://github.com/user-attachments/assets/614fdb26-65ec-400c-a43c-73bfd719d15a" />
<br>This plot shows the resulting profit for a given window of 100 days following the model. The "profit" is **-2%**.<br/>
Here is the actual USD/EUR exchange for that period of time:<br>
<img width="639" height="479" alt="Real stock changes" src="https://github.com/user-attachments/assets/8e0f10da-3f2f-4919-806d-a9d3d834f645" />

Eventhough there is a loss of 2%, this still beats random choice.<br/>

### Usage
Using the provided currency stock file `usd2eur.csv` extracted using GoogleFinance API, we run `python stocks_nlp.py -f "usd2eur.csv" -d <number-of-days>`.

