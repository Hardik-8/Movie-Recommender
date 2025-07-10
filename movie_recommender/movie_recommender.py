
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tkinter as tk

# Load dataset
movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].fillna('')

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Reverse mapping of movie titles to index
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def get_recommendations(title):
    if title not in indices:
        return ["Movie not found!"]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Tkinter UI
def recommend():
    movie = entry.get()
    recommendations = get_recommendations(movie)
    result_var.set("\n".join(recommendations))

# GUI
root = tk.Tk()
root.title("Movie Recommender")

tk.Label(root, text="Enter Movie Name:").pack(pady=5)
entry = tk.Entry(root, width=50)
entry.pack(pady=5)

tk.Button(root, text="Get Recommendations", command=recommend).pack(pady=10)

result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, justify="left", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()
