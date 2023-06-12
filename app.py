from flask import Flask, request, render_template
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.sparse import coo_matrix

app = Flask(__name__)

# Load your book dataframe here (replace 'books.csv' with the actual filename)
books = pd.read_csv('books.csv')
ratings = pd.read_csv("ratings.csv")

# enter the ids the books you like from the search
my_book_ids = []
my_ratings = []

# create a dictionary with book IDs and ratings
data = {
    'user_id': [-1] * len(my_book_ids),  # Set my_id as -1 for all ratings
    'book_id': my_book_ids,
    'rating': my_ratings
}

# create the my_books dataframe
my_books = pd.DataFrame(data)


# Combining title and authors into one column
books["tags"] = books["title"] + " " + books["authors"]

# Cleaning tags
# remove unwanted characters
books["tags"] = books["tags"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
# lower case
books["tags"] = books["tags"].str.lower()
# remove 2+ more spaces and replace with 1 space
books["tags"] = books["tags"].str.replace("\s+", " ", regex=True)
# removing rows that contain no titles.
books = books[books["tags"].str.len() > 0]

# Create the TF-IDF matrix
vecto = TfidfVectorizer()
tfidf = vecto.fit_transform(books["tags"])


# show book front covers
def show_image(val):
    return '<a href="{}"><img src="{}" width=50></img></a>'.format(val, val)


# search function that takes in a user query
def search(q):
    # clean the query
    q_clean = re.sub("[^a-zA-Z0-9 ]", "", q.lower())

    # convert to a vector
    q_vec = vecto.transform([q_clean])

    # cosine similarity between the query and our tfidf matrix
    similarity = cosine_similarity(q_vec, tfidf).flatten()

    # get the indices of the books that are similar to our query, selecting the 10 highest
    indices = np.argpartition(similarity, -10)[-10:]

    # use the indices to get the book titles
    results = books.iloc[indices]

    results["similarity"] = similarity[indices]
    
    results["similarity"] =  results["similarity"].round(3)

    # sort the results. Only show results that have a similarity score greater than 0.2
    results = results[results["similarity"] >= 0.2].sort_values("similarity", ascending=False)

    return results[["authors", "original_title", "average_rating", "image_url", "similarity", "id"]]


######################
#### filter users ####
######################

def filter_users(ratings, mybooks):
    # We want to find users, that rate the books we rated highly also highly. "Common Users"
    # This is to reduce the number of rows in our ratings dataset
    
    
    # set the threshold for high ratings
    high_rating_threshold = 3
    
    mybooks["rating"] = mybooks["rating"].astype(int)
    mybooks["book_id"] = mybooks["book_id"].astype(int)
    mybooks["user_id"] = mybooks["user_id"].astype(int)
    # get the book IDs of the books you rate highly
    highly_rated_books = mybooks.loc[mybooks['rating'] >= high_rating_threshold, 'book_id']

    # filter the ratings dataset to select users who rate the highly rated books
    filtered_ratings = ratings[ratings['book_id'].isin(highly_rated_books)]
    print("filtered_ratings: ", filtered_ratings)

    # get the unique user IDs from the filtered ratings
    selected_users = filtered_ratings['user_id'].unique()

    # filter the ratings dataset to keep only the selected users
    filtered_ratings = ratings[ratings['user_id'].isin(selected_users)]
    
    # we add our ratings to our the set of filtered_ratings
    filtered_ratings = pd.concat([filtered_ratings, mybooks], ignore_index = True)

    return filtered_ratings

#################################
#### create user item matrix ####
#################################

def user_item_matrix(filtered_ratings):
    
    # preprocessing our filtered ratings matrix before converting it to a user_item matrix
    filtered_ratings["user_id"] = filtered_ratings["user_id"].astype(int)
    filtered_ratings["user_index"] = filtered_ratings["user_id"].astype("category").cat.codes
    filtered_ratings["book_index"] = filtered_ratings["book_id"].astype("category").cat.codes
    

    # we will use a sparse matrix rather than a dense matrix to save memory
    ratings_mat_coo = coo_matrix((filtered_ratings["rating"], 
                                  (filtered_ratings["user_index"], filtered_ratings["book_index"])))
    
    # covnerting our matrix to csr format
    ratings_mat = ratings_mat_coo.tocsr()
    
    # finding row position
    #my_index = filtered_ratings[filtered_ratings["user_id"]==-1]["user_index"].values[1]
    
    return ratings_mat

#####################################
#### calculate cosine simularity ####
#####################################

def simularity(ratings_mat, my_index):
    user_similarity = cosine_similarity(ratings_mat[my_index,:], ratings_mat).flatten()
    return user_similarity

#####################################
#### book recommendations system ####
#####################################

def book_rec(ratings, mybooks):
    # drop duplicates
    ratings = ratings.drop_duplicates()
    
    
    

    # re ordering
    ratings = ratings[["user_id", "book_id", "rating"]]
    
    
    # filter the users using the filter users function
    filtered_ratings = filter_users(ratings, mybooks)
        
    # create a user item matrix
    ratings_mat = user_item_matrix(filtered_ratings)
    
    # finding row position
    my_index = filtered_ratings[filtered_ratings["user_id"]==-1]["user_index"].values[1]
    
    # calculate cosine simularity
    user_similarity = simularity(ratings_mat, my_index)
    
    
    #### find simular users based of cosine simularity ####
    
    # find the indicies of the users that are most simular to us and take the first 15
    indices = np.argpartition(user_similarity, -15)[15:]

    # we need to find the user ids
    simular_users = filtered_ratings[filtered_ratings["user_index"].isin(indices)].copy()

    # lets remove ourself from this list
    simular_users = simular_users[simular_users["user_id"] != -1]

    # this dataframe now contains book potential recomendations of users that are most simular to us
    # simular_users
    
    ####
    
    
    ##### finding out how many times each book appears in this list ####

    book_recs = simular_users.groupby("book_id").rating.agg(["count", "mean"])

    # adding the ttitle
    book_recs = book_recs.merge(books[["id", "title", "authors", "ratings_count", "image_url"]], left_index=True, right_on="id")

    # this contains the amount of times each book appears in our rec list and the mean ratings
    #book_recs
    
    ####
    
    #### ranking recomendations ####
    #
    # we need to rank our reccomendations
    #
    # we want to have books that are popular amonst simular users to us. We do not want books that are popular in general
    #
    # so we want to get the books everyone likes, we want the books that only people simular to us liked,
    #
    # and general people didnt like or pay much attention too
    #
    # for example if everyone rates harry potter, and people simular to us rate harry potter. we dont know if 
    #
    # people simular to us like harry potter because of a simular trait or because everyone likes harry potter.


    # creating an adjusted count
    book_recs["relative_count"] = book_recs["count"]/book_recs["ratings_count"]

    # creating a score
    book_recs["score"] = book_recs["mean"] * book_recs["relative_count"]


    # take out books that we have already read
    book_recs =  book_recs[~book_recs["id"].isin(my_books["book_id"])]
    
    
    ####
    
    # only display books that have a mean rating over grater than 4 by simular users
    top_books = book_recs[book_recs["mean"]>=4]
    top_books = top_books.sort_values(by ="score", ascending = False)[:10]
    
    top_books["mean"] = top_books["mean"].round(2)

    print("top_books", top_books)

    #return top_books.style.format({'image_url': show_image})
    return top_books



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search")
def perform_search():
    query = request.args.get("query")
    results = search(query)
    return render_template("results.html", query=query, results=results)


@app.route("/rate", methods=["POST"])
def rate_book():
    global my_books  # Declare my_books as a global variable
    book_id = request.form["book_id"]
    rating = int(request.form["rating"])

    # Save the rating to my_books dataset
    my_books = my_books.append({"user_id": -1, "book_id": book_id, "rating": rating}, ignore_index=True)
    my_books.to_csv("my_books.csv", index=False)

    return render_template("my_books.html", my_books=my_books)


@app.route("/recommend")
def recommend_books():
    #global ratings
    if len(my_books)<5:
        return "Rate more books. You rated " + str(len(my_books)) + " books!"
    else:
        
        recommendations = book_rec(ratings, my_books)
        return render_template("recommendations.html", recommendations=recommendations)



if __name__ == "__main__":
    app.run()