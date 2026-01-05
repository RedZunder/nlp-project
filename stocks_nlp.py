import argparse
from statistics import mean

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC


##Argument parser setup
def setup_parser(parse):
    parse = argparse.ArgumentParser(prog="stocks_nlp",
                                     description="Trump's tweets sentiment analysis for stock prediction",
                                     epilog="Provide a CSV file for the stocks value and run the program")
    parse.add_argument("-f", "--file", help="CSV with stocks",required=True)
    parse.add_argument("-v", "--verbose", help="Show more info", action="store_true")
    parse.add_argument("-vv", "--debug", help="Show a lot of info", action="store_true")
    parse.add_argument("-u", "--update", help="Download nltk packages. *Strongly recommended* "
                                              "for first-time program execution", action="store_true")
    parse.add_argument("-s", "--simple",help="Only prepare the dataframe",action="store_true")
    return parse.parse_args()

parser = argparse.ArgumentParser()
args=setup_parser(parser)


import time                                                 #to measure speed of the program
print("\nImporting libraries...\n")
start=time.time()


#------------------------LIBRARIES------------------------
from nltk.corpus import stopwords                           #stopwords
import re                                                   #regular expressions (regex)
import nltk                                                 #tokenization
import spacy                                                #parts of speech tagging
from spacy import tokenizer
from spacy import displacy
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize      #tokenizing
from nltk.stem import WordNetLemmatizer                     #lemmatizing
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from vaderSentiment.vaderSentiment\
        import SentimentIntensityAnalyzer                   #Sentiment analysis
from tqdm import tqdm                                       #Loading bar
import datetime
import math
#----------------------------------------------------------



#------------------------FUNCTIONS------------------------

##Clean dataframes from unnecessary data
def prepare_tweets(df1:pd.DataFrame,df2:pd.DataFrame):
    #df2.index = df2.index + df1.index.shape[0]
    #Delete unnecessary columns
    cols_to_drop = ["device", "favorites", "retweets", "isFlagged", "isDeleted"]            #useless columns
    df1.drop(df1[df1["isDeleted"] == 't'].index, inplace=True)                              #remove deleted tweets
    df1.drop(columns=cols_to_drop, inplace=True)
    df1.drop_duplicates("text", inplace=True,ignore_index=True)
    df1.dropna(how="any",inplace=True,ignore_index=True)

    cols_to_drop = ["url", "media", "replies_count", "reblogs_count", "favourites_count"]   #useless columns
    df2.drop(columns=cols_to_drop, inplace=True)
    df2.dropna(how="any",inplace=True,ignore_index=True)
    df2.rename(columns={"content": "text","created_at":"date"},inplace=True)

    #Sort and reindex
    df1.sort_values(by=["date"], inplace=True)
    df2.sort_values(by=["date"], inplace=True)

    #Remove url prefix (NLP confuses them as PERSON)
    for i,j in zip(df1.index.values,df2.index.values):
        df1.loc[i,"text"]=df1["text"][i].replace("https","").replace("http","").replace("www","")
        df2.loc[j,"text"]=df2["text"][j].replace("https","").replace("http","").replace("www","")

##Clean stock dataframe from unnecessary data
def prepare_stocks(df:pd.DataFrame):
    if args.verbose or args.debug:
        print(f"\n\nStocks file:\t\t{df.shape}\t{df.columns.values}")

    df["date"] = df["Date"].apply(get_stock_date)
    df.drop(columns=["Date", "Volume", "High", "Low", "Open"], inplace=True)

    change=[]
    prev_i=0
    for i in df["Close"]:
        if i >1.001*prev_i or i<0.999*prev_i:         # 1% threshold
            change.append(math.copysign(1,i-prev_i))
        else:
            change.append(0)
        prev_i=i

    df["change"]=change
    if args.verbose or args.debug:
        print(f"\nStocks file after cleaning:\t{df.shape}\t{df.columns.values}\n")
        print(df.describe())

##Join all tweets from the same day into one text
def group_tweets(df:pd.DataFrame)->pd.DataFrame:
    txt=[]
    twts=[]
    dates=[]
    dff=pd.DataFrame(columns=["text","date"])
    last_date=df.loc[0,"date"]

    for i in range(df.shape[0]):
        txt.append(df.loc[i,"text"])
        if df.loc[i, "date"] != last_date:
            dates.append(df.loc[i,"date"])
            last_date=df.loc[i,"date"]
            twts.append(". ".join(txt))
            txt=[]

    dff["text"]=twts
    dff["date"]=dates
    if args.verbose or args.debug:
        print(f"After grouping by days:\t\t{dff.shape}")
    return dff

def map_stock_changes(df:pd.DataFrame,df2:pd.DataFrame,st:pd.DataFrame):
    # Repeated dates
    ch = []
    for d in df["date"]:
        if d in st.index.values:
            ch.append(st.loc[d, "change"])
        else:
            ch.append(0)
    df["stock"] = ch

    # Unique dates
    ch = []
    for d in df2["date"]:
        if d in st.index.values:
            ch.append(st.loc[d, "change"])
        else:
            ch.append(0)
    df2["stock"] = ch

##Translate string to date
def get_date(strdate)->datetime.date:
    return datetime.datetime.strptime(strdate[0:10], "%Y-%m-%d").date()

##For stock's different format
def get_stock_date(strdate)->datetime.date:
    return datetime.datetime.strptime(f"{strdate[0:5]}/20{strdate[6:8]}", "%m/%d/%Y").date()

##Remove stopwords from the texts in dataframe
def remove_stopwords(df:pd.DataFrame):
    if args.update:
        nltk.download('stopwords')

    en_stopwords = stopwords.words('english')
    yeswords = ['not', 'did', "aren't", 'all', 'who', 'am']  # Words to remove from the stopword list
    for w in yeswords:
        en_stopwords.remove(w)

    en_stopwords.append('go')

    no_stop_tweets = []
    for ss in tqdm(df["text"]):
        no_stop_tw = []
        for word in ss.split():
            if word not in en_stopwords:
                no_stop_tw.append(word)

        no_stop_tweets.append(" ".join(no_stop_tw))

    df["reduced"] = no_stop_tweets
    if args.verbose:
        print("\n",df.head())
    else:
        if args.debug:
            print("\n",df)

##Tokenize texts in the dataframe
def tokenize(df:pd.DataFrame):
    if args.update:
        nltk.download('punkt_tab')

    lowercase = []
    for ss in df["reduced"]:
        lowercase.append(ss.lower())

    df["tokens"] = lowercase
    tokens = []
    for ss in df["tokens"]:
        tokens.append("".join(sent_tokenize(re.sub(r"[^\w\s]", "", ss))))  # remove punctuation

    df["tokens"]=tokens
    if args.verbose or args.debug:
        print(f"\n\nREDUCED:\n{df["reduced"][100]}\nTOKENS:\n{df["tokens"][100]}\n\n")
        if args.debug:
            print(df["tokens"])

##Use lemmatizer on clean text
def lemmatize(df:pd.DataFrame):
    if args.update:
        nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for t in df["tokens"]:
        #lemmas.append(lemmatizer.lemmatize(t).split(" "))
        lemmas.append(lemmatizer.lemmatize(t))

    df["lemmas"] = lemmas
    if args.verbose or args.debug:
        print("\n",df["lemmas"])

##Find organizations or people in the tweets
def find_entities(df:pd.DataFrame):
    nlp = spacy.load('en_core_web_sm')
    orgs_col = []
    for i in tqdm(df["tokens"]):
        spacy_doc = nlp(i)  # analyze tweet
        orgs = []

        for w in spacy_doc.ents:
            if w.label_ == "ORG" or w.label_ == "PERSON" or w.label_ == "GPE":
                orgs.append(w.text_with_ws)

        orgs_col.append(",".join(orgs))
    df["entities"] = orgs_col
    if args.verbose or args.debug:
        print("\n",df["entities"].head())

##Use sentiment analysis on the lemmas
def analyze_sentiment(df:pd.DataFrame):
    vader_sentiment = SentimentIntensityAnalyzer()
    scores = []
    for w in tqdm(df["lemmas"]):
        #scores.append(vader_sentiment.polarity_scores(" ".join(w))["compound"])
        scores.append(vader_sentiment.polarity_scores(w)["compound"])

    df["sentiment"] = scores
    if args.verbose:
        print(df["sentiment"].describe())
    if args.debug:
        print(f"\nData frame:\t{df.columns.values}\n")


#----------------------------------------------------------

#----------------------------------------MAIN CODE----------------------------------------
print(f"Importing libraries - Took {(time.time()-start)*1000} ms\n")
start=time.time()

oil=pd.read_csv(args.file)
tweets1=pd.read_csv('tweets.csv')
tweets2=pd.read_csv('truth_archive.csv')

#------------DATA PREPARATION-------------------
print("\n\t\t-------------------------DATA PREPARATION-------------------------\n")

print(f"oil stocks:\t\t\t{oil.shape}")
print(f"tweets 2009-2021:\t\t{tweets1.shape}\n")
if args.verbose or args.debug:
    print(f"COLUMNS: -------->\t{tweets1.columns.values}\n")
    print(tweets1.head())

print(f"truth social 2022-2025:\t\t{tweets2.shape}\n")
if args.verbose or args.debug:
    print(f"COLUMNS: -------->\t{tweets2.columns.values}\n")
    print(tweets2.head())
    print(f"\n--------Before cleaning:\t{tweets1.shape} {tweets2.shape}\n")

prepare_tweets(tweets1,tweets2)

if args.verbose or args.debug:
    print(f"--------After cleaning:\t\t{tweets1.shape} {tweets2.shape}\n")
    print("\n\t\t-------------------------TWEETS-------------------------\n")
    print(tweets1.head())
    print("\n\t\t-------------------------TRUTH SOCIAL-------------------------\n")
    print(tweets2.head())
    print("\n\t\t------------------------------------------------------\n\n")

if args.verbose or args.debug:
    print(f"COLUMNS: -------->\t{tweets1.columns.values}  ,  {tweets2.columns.values}\n")

dataf=pd.concat([tweets1,tweets2],join="outer")
dataf.set_index(pd.Index(range(dataf.shape[0])), inplace=True)
dataf["date"]=dataf["date"].apply(get_date)


#Group by date
dataf2=group_tweets(dataf)

#Prepare stocks
stocks = pd.read_csv(args.file)
prepare_stocks(stocks)


stocks.set_index(stocks["date"],inplace=True)
if args.verbose or args.debug:
    print(stocks.columns.values)
    print(stocks.index.values)


map_stock_changes(dataf,dataf2,stocks)

print("\n\n")
print(dataf["stock"].describe())
print("\n")
print(dataf2["stock"].describe())
print("\n")

if args.verbose or args.debug:
    print("Columns of price changes:\n")
    print(f"(Not grouped) Not zeros: {dataf.shape[0]-dataf.isin([0]).sum(axis=0)["stock"]}")
    print("\n")
    print(f"(Grouped) Not zeros: {dataf2.shape[0]-dataf2.isin([0]).sum(axis=0)["stock"]}")
    print("\n\n")
# ------------------------------------------

if args.simple:
    exit()

#------------STOPWORDS-------------------

print("\n\t\t-------------------------STOPWORDS-------------------------")
remove_stopwords(dataf)
remove_stopwords(dataf2)

# ------------------------------------------



# ------------TOKENIZATION-------------------

print("\n\t\t-------------------------TOKENIZATION-------------------------")
tokenize(dataf)
tokenize(dataf2)

# ------------------------------------------



#------------LEMMATIZATION-------------------

print("\n\t\t-------------------------LEMMATIZATION-------------------------")
lemmatize(dataf)
lemmatize(dataf2)

#---------------------------------------

#print("\n\t\t-------------------------ENTITIES-------------------------")

#find_entities(dataf)
#find_entities(dataf2)


#---------------------------------------


#------------SENTIMENT ANALYSIS-------------------
print("\n\t\t-------------------------SENTIMENT-------------------------")

analyze_sentiment(dataf)
analyze_sentiment(dataf2)

if args.verbose:
    print(f"\n{dataf.columns.values}")
    print(f"\n{dataf2.columns.values}")
if args.debug:
    print(dataf)

#------------TRAINING MODEL-------------------
print("\n\t\t-------------------------MODEL TRAINING-------------------------")


transf=ColumnTransformer(transformers=[("lemms",CountVectorizer(strip_accents='ascii',min_df=0.01),"lemmas"),
                                  #("ents",CountVectorizer(strip_accents='ascii',min_df=0.01),"entities"),
                                  ("sents",MinMaxScaler(),["sentiment"])])

dff=dataf2.loc[:,["sentiment", "lemmas"]]

X=transf.fit_transform(dff)

targ=dataf2["stock"]

if args.verbose or args.debug:
    print(f"\nTarget: {targ.shape}\t\tDataframe: {X.shape}\n")

#Divide data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X.toarray(), targ, test_size=0.2, random_state=42)
if args.verbose or args.debug:
    print('X_train: ', x_train.shape)
    print('X_test: ', x_test.shape)
    print('y_train: ', y_train.shape)
    print('y_test: ', y_test.shape)


#Test the models

sgd = SGDClassifier(random_state=42).fit(x_train, y_train)
pred=sgd.predict(x_test)
print("\nAccuracy:\t",accuracy_score(pred,y_test))

lr=LogisticRegression(random_state=42).fit(x_train, y_train)
pred=lr.predict(x_test)
print("\nAccuracy:\t",accuracy_score(pred,y_test))

pred=lr.predict(X[4750:4760,:])
print(dataf2["stock"].iloc[4750:4760].values,"\n",pred)
print("\n")
#---------------------------------------------------------------------------------------

dff=[]

dff=dataf.loc[:,["sentiment", "lemmas"]]
X=transf.fit_transform(dff)
targ=dataf["stock"]

if args.verbose or args.debug:
    print(f"\nTarget: {targ.shape}\t\tDataframe: {X.shape}\n")

#Divide data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X.toarray(), targ, test_size=0.2, random_state=42)
if args.verbose or args.debug:
    print('X_train: ', x_train.shape)
    print('X_test: ', x_test.shape)
    print('y_train: ', y_train.shape)
    print('y_test: ', y_test.shape)

#Test the models

sgd = SGDClassifier(random_state=42).fit(x_train, y_train)
pred=sgd.predict(x_test)
print("\nAccuracy:\t",accuracy_score(pred,y_test))

lr=LogisticRegression(random_state=42).fit(x_train, y_train)
pred=lr.predict(x_test)
print("\nAccuracy:\t",accuracy_score(pred,y_test))

pred=lr.predict(X[80050:80060,:])
print(dataf["stock"].iloc[80050:80060].values,"\n",pred)
print("\n")