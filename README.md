# Generate Personalized Reading Recommendations from a Book Dataset

**Task:** You have been given a dataset containing information about various books,
including the title, author, genre, and a brief summary. Your task is to develop a simple
but creative recommendation system using Python to provide personalized reading
recommendations.

## Overview of what this project contains

This project involves building a book recommendation system that incorporates both a search engine and personalized recommendations using collaborative filtering. 

The system allows users to search for books by titles and author names. Users can then rate the books they have read, and create user profiles. 

These user profiles are then used to generate personalized book recommendations using collaborative filtering techniques. The system aims to provide users with relevant book suggestions based on their preferences and the ratings of similar users. 

The system was then deployed onto a web application using Flask. You can run the app.py file and access it by typing http://localhost:5000/ into a web browser.

![image](/screenshots/1home.png)


## Data
Data was from kaggle [goodbooks-10k](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k?select=books.csv).

Contains data on 10k books and 10k users with almost a million rows of user interactions.

# Methods

First a search engine was built. Then we used user-based colaborative filtering to make recommendations. The steps taken is described bellow with code.

## Building a search engine

The aim of this section is to develop a book search engine that enables users to search for books based on titles and author names. 

The search engine will serve as the foundation for creating user profiles, allowing individuals to search for books they like and rate them. 

These user profiles will then be utilized to generate personalized book recommendations using collaborative filtering techniques.

### Steps taken to build search engine

**1. Preprocess Text Data**

Cleaned the text data by removing special characters, large spaces, and converting to all text to lower case.  

**2. Term Frequency Matrix (TF):**

Constructed a term frequency matrix to capture the frequency of each term within each book title and author columns.
    
    
**3. Inverse Document Frequency Matrix (IDF):**

Constructed an inverse document frequency matrix to minimize the impact of common words such as "the" or "and" that appear frequently across books.

**4. TF-IDF Calculation:**

Combined the term frequency matrix and the inverse document frequency matrix to obtain a TF-IDF matrix.

**5. Searching with TF-IDF Vectors:**

Enabled users to enter search queries (book titles or author names) and converted them into TF-IDF vectors using the same term frequency and inverse document frequency calculations.

Determined the similarity between the TF-IDF vector of the search query and the TF-IDF vectors of the books in the collection using cosine simularity.

Ranked the books based on their cosine similarity to the search query and presented the results to the user.

**6. Create a jupyter notebook widget to search for books**

Make a notebook widget to allow users to search for books they want.

**How to use:**
Just enter the name of an author or a book title. You dont need to know the whole auther name or even the entire book name. You can enter partial names and partial titles. Or even a combination of both. Names and titles have to be spelled correctly.

## Memory-based collaborative filtering 


Memory-based collaborative filtering relies on the user-item interaction data to make recommendations. 

For this coding task user-item filtering was used. A user-item filtering will take a particular user, find users that are similar to that user based on similarity of ratings, and recommend items that those similar users liked.

You can say: “Users who are similar to you also liked …”

### Steps taken to implement collaborative filtering

**1. User Filtering based on Books Read:**

Filter out users who have read the same books as us and have given high ratings to the books we rated highly. This step aims to identify users with similar preferences and reduce the dimensions of the user-item matrix.

**2. User-Item Matrix Creation:**

Construct a user-item matrix based on the filtered set of users. This matrix represents the ratings given by users to different books, enabling comparisons and analysis.

**3. Similar User Identification:**

Calculate the similarity between users based on their rating patterns. Used cosine similarity to identify users who have similar preferences.

**4. Book Ratings by Similar Users:**

Extract the books that similar users have rated. Determine the count of each book and calculate the average rating given by the similar users.

**5. Adjusted Count for Personalization:**

Focus on books that are highly rated by similar users but may not be popular among all users. Adjust the count of user ratings by considering the proportion of similar users who have rated each book.

**6. Scoring and Sorting of Recommendations:**

Score the book recommendations by multiplying the adjusted count by the mean rating given by similar users. Sort the recommendations based on this score to present personalized book recommendations to the user.

**7. Evaluation:**

To evaluate the peformance of the system a simple manual check is done to see if recomendations are appropriate.

## Limitations and how they could be improved

**1. Experiment with different filtering methods**

It would have been better to explore different filtering methods, such as content-based filtering and other types of collaborative filtering like item-item collaborative filtering.

It would have also been interesting to explore other techniques like model-based collaborative filtering, which is based on matrix factorization. A common matrix factorization method is singular value decomposition, which is what the famous winning team at the Netflix Prize competition used to make recommendations.

**2. Better System Evaluation**

A better and more systematic evaluation method should have been utilized to measure the performance of the system. To gain a better understanding of how the system performs, the data could have been split into training and testing sets. It is common in research to use metrics like mean squared error (MSE) or root mean squared error (RMSE) to evaluate the performance of recommendation systems.

**3. Cold Start Problems**

The system faces a cold start problem when new users or users without ratings join. The current solution of using a search engine for users to rate books helps, but the problem persists for users who haven't rated anything or for new books. To tackle this, a hybrid approach combining content-based and user-based filtering could be implemented. Content-based filtering considers book attributes, which helps with new users, while user-based filtering relies on existing user data. This hybrid approach improves recommendations even in cold start situations.

**4. Dataset**

In the future, I would like to use a better dataset. This dataset only contains 10k books. Some dataset contain millions of books. That would have been much more intresting to work with. However I used this smaller dataset because of RAM limitations. 


I am sure there are many more limitations and ways to improve on this, these are just some of the ways I could think off right now. I think the biggest limitation is the peformance evaluation. Because no quantitative measure was used to evaluate the peformance of the system.

**What I have Learned**
- The basics of recommender systems and some of their algorithms.
- First time to use Flask and learned some basic HTML.


