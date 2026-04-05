# 🎬 MovieMatch: A Vector-Based Recommendation Engine

MovieMatch is a full-stack Machine Learning application that suggests movies based on user rating patterns. Unlike simple genre-based filtering, this engine uses **Collaborative Filtering** logic to find "hidden" similarities between films based on how thousands of users have rated them.



## 🧠 Technical Architecture 
The core of this project is built on the **K-Nearest Neighbors (KNN)** algorithm.

- **Algorithm:** Unsupervised `NearestNeighbors`.
- **Distance Metric:** **Cosine Similarity**. 
  - *Why Cosine?* In high-dimensional rating spaces, Euclidean distance can be skewed by "lenient" vs "strict" raters. Cosine Similarity measures the **angle** between movie vectors, focusing on the pattern of ratings rather than the absolute values.
- **Data Optimization:** Used `scipy.sparse.csr_matrix` to transform a massive User-Item pivot table into a memory-efficient sparse matrix. This allows the app to run fast even with 100,000+ ratings.

## 🛠️ Features
- **Modern UI:** Glassmorphic dark-mode interface built with **Streamlit** and custom **CSS**.
- **Live Metadata:** Integrates with the **OMDb API** to fetch real-time movie posters and data.
- **Smart Search:** Regex-powered search that handles titles and release years independently for higher API accuracy.
- **High Performance:** Implemented `@st.cache_resource` to ensure the ML model loads once and provides instant recommendations.

## 📂 Project Structure
```text
├── app.py                # Streamlit Web Application
├── movie_model.pkl       # Trained KNN Model
├── movie_pivot.pkl       # Processed User-Item Matrix
├── requirements.txt      # Project Dependencies
└── notebooks/
    └── KNN_Training.ipynb # Data cleaning & Model Training logic
