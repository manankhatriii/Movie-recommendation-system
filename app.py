import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib

df = joblib.load('df.pkl')
df_copy = joblib.load('df_copy.pkl')

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

cosine_sim = cosine_similarity(df_scaled, df_scaled)

def get_similar_movies(movie_name, cosine_sim=cosine_sim, df=df, df_copy=df_copy, top_n=10):
    df_copy['Title_lower'] = df_copy['Title'].str.lower()
    movie_name_lower = movie_name.lower()
    
    if movie_name_lower not in df_copy['Title_lower'].values:
        return f"Movie '{movie_name}' not found in the dataset."
    df_copy.reset_index(drop=True, inplace=True)
    
    movie_index = df_copy[df_copy['Title_lower'] == movie_name_lower].index[0]
    sim_scores = list(enumerate(cosine_sim[movie_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    
    similar_movies = df_copy.loc[movie_indices].copy()
    similar_movies['similarity'] = similarity_scores
    similar_movies = similar_movies.drop(columns=['Title_lower'])
    return similar_movies

st.title('MOVIE RECOMMENDATION SYSTEM')

movie_titles = df_copy['Title'].drop_duplicates().tolist()
selected_movie = st.selectbox('Select a movie:', movie_titles)

if selected_movie:
    similar_movies_df = get_similar_movies(selected_movie)
    st.write(f"Movies similar to '{selected_movie}':")
    st.dataframe(similar_movies_df)
