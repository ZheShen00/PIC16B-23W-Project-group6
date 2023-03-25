#!/usr/bin/env python
# coding: utf-8

# # Prject(Group 6) Prediction of Movies' Performance based on IMDb data
# 
# ## ——Visualization and Exploratory data analysis

# In[1]:


import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Checking the dataframe and choosing features
# First, we import the cleaned dataset. The whole dataset has 12 features as well as over 400,000 movie data.

# In[2]:


imdb = pd.read_csv('imdb.csv')
imdb


# In our program, ratings are the most important indicator of a film's popularity. So we can start by looking at the specific data of the ratings.

# In[3]:


imdb.averageRating.describe()


# In our data set, we can see that the average rating is around 6.0 and the median is 6.2. And most scores are between 5.1 and 7.1. This will affect our next specific rating of the movie's popularity. For now, we will make the following scale of popularity: 
# - Very Positive(>7.1)
# - Mostly Positive(6.0-7.1)
# - Mostly nagative(5.1-6.0)
# - Very Nagative(<5.1)
# 
# Then, let's explore which factors are strongly correlated with the scores, which will greatly help our machine learning model to make predictions:

# In[4]:


# Check how many factors our data set have
imdb.columns.values.tolist()


# It can be clearly observed that the first 3 factors are only the number of the movie or the actor/director. So we can exclude these 3 between them.It can be clearly observed that the first 3 factors are only the number of the movie or the actor/director. So we can exclude these 3 between. And averageRating, numVotes, isAdult, startYear and runtimeMinutes are very good quantifiable data. genres would also be one of the factors that could have a significant impact. So we will analyze these factors next.
# 
# First is the distribution of average rating

# In[5]:


def dis_averageRate(data):
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data['averageRating'])
    plt.title('The distribution of  average rating', fontsize=20, weight='bold', color='black')
dis_averageRate(imdb)


# ## The relationship of ratings and other element
# 

# In[6]:


def dis_RateVotes(data):
    plt.figure(figsize=(10, 8))
    plt.scatter(data['numVotes'], data['averageRating'], s=10, c='red')
    plt.title('The distribution of Rating and number of votes', fontsize=16, weight='bold')
    plt.show()
dis_RateVotes(imdb)


# Regarding the relationship between the number of votes and the ratings, the scatterplot presents an uncorrelated feature. However, we can observe that a large number of votes occurs around a rating of 8.0. This means that good movies are more likely to receive a large number of ratings. But the number of votes is statistically uncorrelated for ratings.

# In[8]:


def dis_RateAldut(data):
    plt.figure(figsize=(8, 8))
    sns.boxplot(x="isAdult", y="averageRating", data=data, linewidth=1.5)
    plt.title('The distribution of Rating and Aldut/Not Aldut',fontsize=20, weight='bold')
    plt.show()
dis_RateAldut(imdb)


# boxplot shows that whether a movie is adult-rated or not makes big difference in terms of rating. But not adult movie got higher mean value, so when we doing the machine learning model, this factor would be considered

# In[9]:


def dis_ReleaseYear(data):
    df = data.startYear.value_counts()
    x = [i for i in df.keys()]
    y = []
    for i in range(0,11):
        y.append(df.values[i])
    plt.figure(figsize=(10, 6))
    plt.title('The distribution of film release year', fontsize=20, weight='bold')
    sns.barplot(x=x, y=y)
dis_ReleaseYear(imdb)


# In[10]:


def dis_RateRelease(data):
    plt.figure(figsize=(20, 8))
    sns.boxplot(x="startYear", y="averageRating", data=data, linewidth=1.5)
    plt.title('The distribution of Rating and release year',fontsize=20, weight='bold')
    plt.show()
dis_RateRelease(imdb)


# We can see that people generally have better ratings for new movies released after 2021. Despite the huge impact of the pandemic on the movie industry in 2020 resulting in a drop in the number of releases, we still have 35,000 data per year(except 2023), so it is still a valid conclusion that the newer the movie, the more likely it is to receive a high rating.

# In[17]:


def Split_genres(data):
    global df1
    df1 = data
    df1 = df1.drop(['genres'], axis=1).join(df1['genres'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('genres'))
    return df1
Split_genres(imdb)


# In[18]:


def dis_genres(data):
    df2 = df1.genres.value_counts()
    x = [i for i in df2.keys()]
    y = []
    for i in range(0,28):
        y.append(df2.values[i])
    plt.figure(figsize=(26, 8))
    plt.title('The distribution of genres', fontsize=30, weight='bold')
    sns.barplot(x=x, y=y)
dis_genres(df1)


# In[19]:


def dis_RateGenre(data):
    plt.figure(figsize=(30, 8))
    sns.boxplot(x="genres", y="averageRating", data=data, linewidth=1.5)
    plt.title('The distribution of Rating and genres',fontsize=20, weight='bold')
    plt.show()
dis_RateGenre(df1)


# Here, we can see that Horror, Sci-Fi, Mystery and Western movies are less likely to get high ratings. While Documentary, Drama, Crime, Animation, Musical, History, Biography titles are most likely to get high ratings (except for some genres with too small a sample size).

# ## Summary and future work
# 
# Based on the above analysis, we will make movie rating predictions mainly for genres and release years. In the next work, we will use multiple linear regression models, decision tree regression models, random forest regression models, etc. to predict movie ratings.

# In[30]:


def drop_Column(data):
    drop_columns = ["Unnamed: 0", "tconst", "nconst", "category", "primaryTitle", "primaryName"]
    global df_model
    df_model = data.drop(drop_columns, axis=1)
    return df_model
drop_Column(df1)


# In[27]:


df_model.info()


# In[31]:


def replace_N(data):
    data['genres'] = data['genres'].replace({r'\N': None})
    data['runtimeMinutes'] = data['runtimeMinutes'].replace({r'\N': None}).astype(float)
    return df_model
replace_N(df_model)


# In[33]:


df_model.info()


# In[34]:


df_model = df_model.dropna()


# In[35]:


df_model = df_model.reset_index(drop=True)


# In[36]:


from sklearn.preprocessing import LabelEncoder

# creating instances of labelencoder
labelencoder = LabelEncoder()

# Assigning numeric values and convert the non-numeric column
df_model['genres'] = labelencoder.fit_transform(df_model['genres'])
df_model.head()


# In[37]:


y = df_model['averageRating']
x = df_model.drop(['averageRating'], axis = 1)


# In[38]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# In[41]:


from sklearn.ensemble import RandomForestRegressor

def RFR_model(train_data1,train_data2):
    regr = RandomForestRegressor()
    regr.fit(train_data1,train_data2)
    return regr.score(X_test, y_test)
RFR_model(X_train,y_train)


# In[42]:


from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
def LR_model(train_data1,train_data2):
    model = LinearRegression()
    model.fit(train_data1,train_data2)
    predict_valid = model.predict(X_test)
    score = r2_score(y_test,predict_valid)
    return score
LR_model(X_train, y_train)

