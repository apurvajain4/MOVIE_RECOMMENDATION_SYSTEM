import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st
import requests

df = pd.read_csv("../data/content-based-dataset.csv")

# Understanding the need for the column "tags"¶
# We recommend movies to the user based on the similarity of the user-inputed-movies and the "tags" column.
# It looks something like this-

# 1. Prompt the users with something like this- "Please enter a movie that you like."
# Movie: (user_selected)

# 2. We search the inputed movies'
# genre,
# cast,
# crew,
# keywords.

# 3. We then combine all the values into one single tag. Let's call it query_tag
# Then, we need to find 10 movies which have the most similarity to the query_tag
# Then we derive the similarity between the query_tag and the other tags by converting them into vectors and calculating their cosine_similarity.

from ast import literal_eval
df["genres"] = df["genres"].apply(lambda x: literal_eval(x))


def get_unique(list_of_lists):

    elements = []
    for list1 in list_of_lists:
        for element in list1:
            elements.append(element)
    return list(set(elements))

# UNIQUE GENRES
unique_genres = get_unique(df["genres"])

# UNIQUE TITLES
unique_titles = list(df["title"].unique())


# Converting specific datapoints to vectors
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000) # Selecting top 5k most occuring words

# cv.fit_transform(df["tags"]) ---> this will return us a sparse matrix because most values are 0.
# So, scipy defaults this as a sparse matrix.
# So, we need to convert it into an array.

vectors = cv.fit_transform(df["tags"]).toarray()


from sklearn.metrics.pairwise import cosine_similarity
cos_similarity = cosine_similarity(vectors)


def recommend(movie, user_genre="", min_rating = 1):
    movie_index = df[df["title"] == movie].index[0]
    similarities = cos_similarity[movie_index]
    
    # in order to not lose the movie_index, we can use enumerate function and typecast it to list to make it readable
    recommendations_indexes = sorted(list(enumerate(similarities)), reverse=True, key=lambda x:x[1])
    # In the above code, the "key=lambda x:x[1]" is used to specifically sort the movies based on the similarities and not based on the index values.
    
    recommendations   = []
    similarity_scores = []
    vote_averages   = []
    genres = []
    id_s = []
    for tuples in recommendations_indexes:
        
        id_s.append(df["id"][tuples[0]])
        recommendations.append(df["title"][tuples[0]])
        similarity_scores.append(round(tuples[1], 2))
        vote_averages.append(df["vote_average"][tuples[0]])
        genres.append(df["genres"][tuples[0]])
    
    recommended_df = pd.DataFrame({"id": id_s,
                                   "Titles": recommendations, 
                                   "Similarity": similarity_scores,
                                   "Genres": genres,
                                   "Rating": vote_averages})
    

    try:
        
        if not user_genre:
            
            final_recommendation = recommended_df[recommended_df["Rating"] >= min_rating]
            
            if len(final_recommendation) > 11:
                return final_recommendation[1:11]
            else:
                return final_recommendation[1:]
            
        else:
            # Recommending movies based on genre.
            matching_genres = recommended_df[recommended_df["Genres"].apply(lambda x: user_genre in x)].index
            filtered_recommendation = recommended_df.iloc[matching_genres]

            final_recommendation = filtered_recommendation[filtered_recommendation["Rating"] >= min_rating]

            if len(final_recommendation) > 11:
                return final_recommendation[1:11]
            else:
                return final_recommendation[1:]
    except:
        message = "Sorry. No movies exist with the filters selected. Try changing the filters again."
        return message
    


st.header("Movie Recommender System")

selected_movie = st.selectbox(index=0, label= "Please enter a movie that you like", options= unique_titles)
selected_genre = st.selectbox(index=0, label= "Genre", options= unique_genres)
minimum_rating = st.number_input(args=None, label= "Rating")

# GET THE POSTERS
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

# RECOMMENDATIONS
recommend_button = st.button(label="Recommend")
recommendations = recommend(movie=selected_movie, user_genre=selected_genre, min_rating=minimum_rating).reset_index(drop=True)

def list_to_string(any_list):

    all_genres = [", ".join(x) for x in any_list] 
    return all_genres

recommendations["Genres"] = list_to_string(recommendations["Genres"])

if recommend_button:
    # st.table(data=recommendations)

    st.write("")
    st.write("")
    st.write("")
    st.header("Recommended Movies for you")
    
    titles    = recommendations["Titles"]
    img_files = [fetch_poster(id) for id in recommendations["id"]]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(titles[0])
        st.image(img_files[0])
    with col2:
        st.text(titles[1])
        st.image(img_files[1])

    with col3:
        st.text(titles[2])
        st.image(img_files[2])
    with col4:
        st.text(titles[3])
        st.image(img_files[3])
    with col5:
        st.text(titles[4])
        st.image(img_files[4])

    st.write("")
    st.write("")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(titles[5])
        st.image(img_files[5])
    with col2:
        st.text(titles[6])
        st.image(img_files[6])

    with col3:
        st.text(titles[7])
        st.image(img_files[7])
    with col4:
        st.text(titles[8])
        st.image(img_files[8])
    with col5:
        st.text(titles[9])
        st.image(img_files[9])
else:
    pass

