# PIC 16B group 6 project proposal

## Contributors:
 - [Qiu, Joanne](https://github.com/joanneqiu07)
 - [Shen, Zhe](https://github.com/ZheShen00)
 - [Zhou, Zhitong](https://github.com/Zzzztong)

## Abstract:
In this project, we will build an algorithm to predict whether an upcoming movie will be commercially successful or not. 

- Firstly, we will use **SQL** and **Python** to collect and clean the data.
- Secondly, we will use **visualization** and **data analysis method** to find the characters of commercially successful movies or what factors contribute to their success. 
- Then we will build a **machine learning model** and use the dataset for training and testing. The test results will be compared to verify the accuracy of the model. 
- Finally, we will try to use our algorithm to predict several upcoming movies.

GitHub repository link: https://github.com/ZheShen00/PIC16B-23W-project-group6

# Planned Deliverables
**Full Success**: We will be able to provide an algorithm that allows users to enter information about a movie and provides a predicted value for that movie's rating on [IMDb](https://www.imdb.com/).

**Partial success**: We will provide an exploratory analysis of the factors that can influence the success of a film. It may be difficult for us to do the predictive rating part, but we will demonstrate our core code on Jupyter Notebook and use machine learning models to complete tests on a small number of old movie ratings.

# Resources Required
We found 3 datasets or websites that could provide a large amount of data, the links are under below：


1. https://imerit.net/blog/13-best-movie-data-sets-for-machine-learning-projects-all-pbm/

2. https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

3. https://www.lafabbricadellarealta.com/open-data-entertainment/



We will make adjustments as the project progresses as to how we use the data and whether we need to find new datasets.

# Tools and Skills Required
The tools and skills we required are:

- Flask (developing web app)
- Scrapy (web scraping)
- Matplotlib/Seaborn/Plotly (data visualization)
- Sklearn (data preprocessing, creat confusion matrix...)
- Pandas/Numpy/Sqlite3 (data analysis)
- Pytorch (machine learning model package)
- GitHub (team collaboration)

We will add additional necessary tools and skills in the project progresses.

# What We Will Learn 
1.   Technically, we will learn to use different packages in Python and potentially use Python to web scrape some of the data we need. Specifically, we will use SQL and database to clean and organize the data, use Pytorch to create machine learning models, use Plotly to visualize the data, and use PCA and pipeline to adjust the datasets to solve potential bias or problems that may affect the models.
2.   Above all, we will learn to conduct a complex research on movies industry and gain a deep understanding of the relationship between movies and popularity, especially people’s preference on movies nowadays, features of movies that affect their box office and reviews/scores, and the trend of movies production in current years.

# Risks

The potential risks of this project might be:
1. There are many unpredictable factors such as natural disasters, pandemics, and global events, which can impact a movie's success. For example, a pandemic could lead to theater closures, reducing the box office revenue of a movie. Moreover audience preferences change over time, and a movie that may have been successful in the past may not be well-received in the present.
2. We choose to use IMDB score as the determine factor to measure a movie's success. However, IMDb ratings are based on user ratings, which can be influenced by various factors such as personal preferences, biases, and user engagement with the site, so it may introduce a self-selection bias. For example, users who are more likely to rate a movie are likely to be those who have strong opinions about it, whether positive or negative. This can skew the ratings and make them less representative of the general audience's response to the movie. IMDb ratings can also be influenced by timing, as they may change over time as new ratings and reviews are added. This can make it difficult to determine the "true" rating of a movie and may not accurately reflect its long-term success. All these potential factors can affect our intending deliverables and make the model inaccurate.
3. There might be overfitting issues, overreliance on past data can lead to overfitting the model to the training data and poor predictions on new, unseen data. Overfitting occurs when a model is too closely fit to the training data and does not generalize well to new data. This can lead to the model having a high accuracy on the training data but a low accuracy on new data.



# Ethics

1. Our models can reinforce or amplify existing biases and discrimination in the movie industry, such as gender, race, or socioeconomic status. For example, a model that is trained on historical data may produce results that unfairly favor certain types of movies or talent, leading to fewer opportunities for underrepresented groups. It can also potentially reduce creativity and experimentation in the movie industry by prioritizing commercial success over artistic expression. This can result in a homogenization of movie content and limit the diversity of voices and perspectives that are represented on screen.
2. We choose to use IMDB score as the determining factor to measure a movie's success. However, IMDb ratings are based on user ratings, which can be influenced by various factors such as personal preferences, biases, and user engagement with the site, so it may introduce a self-selection bias. For example, users who are more likely to rate a movie are likely to be those who have strong opinions about it, whether positive or negative. This can skew the ratings and make them less representative of the general audience's response to the movie. IMDb ratings can also be influenced by timing, as they may change over time as new ratings and reviews are added. This can make it difficult to determine the "true" rating of a movie and may not accurately reflect its long-term success. All these potential factors can affect our intending deliverables and make the model inaccurate.
3. There might be overfitting issues in our model, overreliance on past data can lead to overfitting the model to the training data and poor predictions on new, unseen data. Overfitting occurs when a model is too closely fit to the training data and does not generalize well to new data. This can lead to the model having a high accuracy on the training data but a low accuracy on new data.


**What groups of people have the potential to benefit from the existence of our prediction?**

Movie investors can benefit from our model because they can use our prediction to make informed decisions about which movie to produce and invest in. By using predictive models, they can identify which movies are likely to be commercially successful and make data-driven investment decisions.

**What groups of people have the potential to be harmed?**

Independent filmmakers have the potential to be harmed by our model because our prediction may prioritize the commercial success of a movie over the artistic and creative expression, thus leading to a decrease in support for independent filmmakers and more mainstream, formulaic films. Also the whole movie industry might be harmed if this model is widely used because The heavy reliance on predictive models may result in a decrease in creativity and originality in the movie industry, as movie studios and investors prioritize safe and predictable investments over more risky and innovative projects.

# Tentative Timeline
## Week 4
*   Discuss the topic and construct the initial proposal

## Week 5 
*   Conduct literature review about movies and popularity
*   Decide on what varibles to choose

## Week 6
*   Data engineering: data collection, cleaning, imputation
*   Find the correlation among the variables and decide on which variables to use in the models
*   Visualize the data
*   **Project update presentation I**

## Week 7
*   Train a clustering ML model to cluster movies in 4-5 groups
*   Analyze their features


## Week 8

*   Train a regression model to calculate the scores of different movies
*   Report different scores: accuracy, f-1, recall, etc.
*   Adjust the model according to different scores and may use PCA to adjust the data set


## Week 9
*   Predict 2-3 movies' popularity by utilizing the two models to analyze their features and calculate their scores
*   **Project update presentation II**


## Week 10
*   Conduct report
*   Analyze the strength and weakness of this project
*   Further discussion on movies and popularity

