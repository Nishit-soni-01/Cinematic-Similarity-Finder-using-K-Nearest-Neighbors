import os
import streamlit as st
import pickle
import pandas as pd
import requests
import numpy as np
import re

# --- 1. CONFIG MUST BE FIRST ---
st.set_page_config(page_title="Movie Matcher Pro", layout="wide", page_icon="🎬")

# --- 2. CUSTOM STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    div.stButton > button:first-child {
        background-color: #e50914;
        color: white;
        border-radius: 5px;
        border: none;
        height: 3em;
        width: 100%;
        font-weight: bold;
        font-size: 18px;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #ff0a16;
        border: none;
        color: white;
        transform: scale(1.02);
    }
    /* Poster hover effect */
    img {
        border-radius: 10px;
        transition: 0.3s;
    }
    img:hover {
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. API LOGIC ---
OMDB_API_KEY = "736f735"

def fetch_poster(movie_title):
    try:
        movie_name = movie_title
        year = None
        match = re.search(r'\((.*?)\)', movie_title)
        if match:
            year = match.group(1)
            movie_name = movie_title.split(' (')[0]

        url = f"http://www.omdbapi.com/?t={movie_name}&y={year}&apikey={OMDB_API_KEY}"
        data = requests.get(url).json()
        
        if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
            return data['Poster']
        
        url_fallback = f"http://www.omdbapi.com/?t={movie_name}&apikey={OMDB_API_KEY}"
        data_fallback = requests.get(url_fallback).json()
        return data_fallback.get('Poster', "https://via.placeholder.com/500x750?text=Poster+Not+Found")
        
    except Exception:
        return "https://via.placeholder.com/500x750?text=Service+Error"

# --- 4. LOAD DATA ---
@st.cache_resource
def load_assets():
    # Get the directory where app.py is located
    base_path = os.path.dirname(__file__)
    
    # Create absolute paths to the pkl files
    model_path = os.path.join(base_path, 'movie_model.pkl')
    pivot_path = os.path.join(base_path, 'movie_pivot.pkl')
    
    # Load the files using the absolute paths
    model = pickle.load(open(model_path, 'rb'))
    movie_pivot = pickle.load(open(pivot_path, 'rb'))
    
    return model, movie_pivot
try:
    model, movie_pivot = load_assets()
    movie_list = movie_pivot.index.values
except Exception as e:
    st.error("Error: 'movie_model.pkl' or 'movie_pivot.pkl' not found.")
    st.stop()

# --- 5. MAIN UI ---
st.title("🎬 Movie Similarity Engine")
st.markdown("Discover movies based on rating similarities using **K-Nearest Neighbors**.")
st.markdown("---")

with st.sidebar:
    st.header("How it works")
    st.info("This app uses **Cosine Similarity** to find users with similar tastes.")
    st.write("---")
    st.caption("Developed as a Portfolio Project")

# Search Section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    selected_movie = st.selectbox("Search or select a movie:", movie_list)
    recommend_btn = st.button('Get Recommendations')

# --- 6. RECOMMENDATION DISPLAY ---
if recommend_btn:
    movie_idx = np.where(movie_pivot.index == selected_movie)[0][0]
    distances, suggestions = model.kneighbors(
        movie_pivot.iloc[movie_idx, :].values.reshape(1, -1), 
        n_neighbors=6
    )
    
    st.subheader(f"Because you liked '{selected_movie}':")
    st.markdown("<br>", unsafe_allow_html=True)
    
    cols = st.columns(5)
    for i in range(1, 6):
        suggested_title = movie_pivot.index[suggestions[0][i]]
        poster_url = fetch_poster(suggested_title)
        
        with cols[i-1]:
            # Use 'use_column_width' for older Streamlit or 'use_container_width' for newer
            st.image(poster_url, use_container_width=True)
            st.markdown(f"**{suggested_title}**")
