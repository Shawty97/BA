import pandas as pd
import numpy as np

import datetime

from sklearn.multiclass import OneVsRestClassifier

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, confusion_matrix


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTENC


#read the dataset
raw_data = pd.read_csv("C:/Users/Admin/Desktop/git/BA/src/data/companies_status.csv", index_col=0)
print(raw_data)

#defining important functions for feature extraction.


#function for extracting mean from the string lists..
def mean(x):

  #takes string list from the dataframe and strips it off of "[]" from both ends, 
  #and then splits the string by ","
  str_list = x.strip("[]").split(",")

  #the result of the line above is a list with string digits in it instead of int or float,
  #so we use eval method on the str_list using the list comprehension technique.
  int_list = [eval(i) for i in str_list]

  #finally calculate the mean and return it.
  mean = sum(int_list)/len(int_list)

  return mean




#function for extracting counts from the string lists..
def count(x):
  str_list = x.strip("[]").split(",")

  int_list = [eval(i) for i in str_list]

  count = len(int_list)

  return count

#extract new features from old ones.

raw_data["no_of_tweets_per_company"] = raw_data['pos_tweet_count'] + raw_data['neg_tweet_count'] + raw_data['neutr_tweet_count']
raw_data["avg_retweets_count"] = raw_data["retweet_counts"].apply(lambda row: mean(row))
raw_data["avg_like_count"] = raw_data["like_counts"].apply(lambda row: mean(row))
raw_data["avg_reply_count"] = raw_data["reply_counts"].apply(lambda row: mean(row))
raw_data["avg_vader_vector"] = raw_data["sent_vader_vector"].apply(lambda row: mean(row))

# pandas is treating dates as string objects, so we need to change that to datetime format.
# look at the columns "founded_on" and "last_funded_on" etc..
print(raw_data.dtypes)


# we use pandas' to_datetime function to change string objects to datetime and then from datetime
# to simple float timestamps.
raw_data["founded_on"] = pd.to_datetime(raw_data["founded_on"]).apply(lambda v: v.timestamp())
raw_data["first_funding_on"] = pd.to_datetime(raw_data["first_funding_on"]).apply(lambda v: v.timestamp())
raw_data["last_funding_on"] = pd.to_datetime(raw_data["last_funding_on"]).apply(lambda v: v.timestamp())

# we can see that all three columns are not string objects anymore but float
print(raw_data.dtypes)

#drop unnecessary and repetetive columns

raw_data.drop(["sent_vader_vector", "retweet_counts", "like_counts", "reply_counts", "Unnamed: 14", "Unnamed: 15", "funded_2012"], axis=1, inplace=True)

print(raw_data.shape)

raw_data.to_csv("data/companies_status_preprocessed.csv")


from sklearn.preprocessing import LabelEncoder
Lenc = LabelEncoder()
raw_data["status"] = Lenc.fit_transform(raw_data.status)


#Features
data = raw_data[[
                 "founded_on", "first_funding_on", "last_funding_on", 
                 "no_of_tweets_per_company", "pos_tweet_count", "neg_tweet_count","neutr_tweet_count", 
                 "avg_retweets_count","avg_like_count","avg_reply_count", 
                 "status",
                 "total_funding",
                 "num_funding_rounds",
                 ]]


#specifying target and features.
y = data.status
x = data.drop("status", axis=1)

#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


#making a preprocessing pipeline so if the model is given a categorical feature, it can handle it.
preprocessor = make_column_transformer(
    (OneHotEncoder(sparse=True, handle_unknown='ignore'), make_column_selector(dtype_include=object)),
    remainder='passthrough'
)

#for debugging purposes 
preprocessor.fit_transform(x_train)
preprocessor.fit_transform(x_train).shape
print(preprocessor.fit_transform(x_train).shape)

#XGBoostClassifier
model = XGBClassifier(n_estimators=200, learning_rate=0.2)
#defining our pipeline with preprocessor and model in it..
pipeline = make_pipeline(preprocessor, model)
#training our model
pipeline.fit(x_train, y_train)
#making prediction and storing it in y_hat
y_hat = pipeline.predict(x_test)
print(y_hat)
#accuracy score
print(accuracy_score(y_test, y_hat))

# as we know we have a huge class imbalance in the dataset
# so the accuracy might not be the best metrics to use. so i am using f1 score with it too.. 
print(f1_score(y_test, y_hat, average='weighted'))



# Random Forest
model = RandomForestClassifier(200)
pipeline = make_pipeline(preprocessor, model)
pipeline.fit(x_train, y_train)
print(pipeline.fit(x_train, y_train))
y_hat = pipeline.predict(x_test)

print(y_hat)
#accuracy score & f1 score
print(accuracy_score(y_test, y_hat))
print(f1_score(y_test, y_hat, average='weighted'))

#SVM
model = SVC(kernel='rbf')
pipeline = make_pipeline(preprocessor, model)
print(pipeline.fit(x_train, y_train))
y_hat = pipeline.predict(x_test)

print(y_hat)
#accuracy score & F1 score
print(accuracy_score(y_test, y_hat))
print(f1_score(y_test, y_hat, average='weighted'))

#Logistic Regression
model = LogisticRegression(solver='lbfgs', max_iter=10000)
pipeline = make_pipeline(preprocessor, model)
print(pipeline.fit(x_train, y_train))
y_hat = pipeline.predict(x_test)

print(y_hat)
#accuracy score & f1 score
print(accuracy_score(y_test, y_hat))
print(f1_score(y_test, y_hat, average='weighted'))

#Naive Bayes
model = GaussianNB()
pipeline = make_pipeline(preprocessor, model)
print(pipeline.fit(x_train, y_train))
y_hat = pipeline.predict(x_test)

print(y_hat)
#accuracy score & f1 score
print(accuracy_score(y_test, y_hat))
print(f1_score(y_test, y_hat, average='weighted'))
