# 🎬 MovieStream AI

A hybrid movie recommendation system that blends **content-based filtering** (TF-IDF on genres) with **collaborative filtering** (SVD on the user-item ratings matrix), served through an interactive **Streamlit** web app.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.6.0-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

MovieStream AI recommends movies by combining two complementary signals:

- **Content-Based Filtering** — represents each movie's genres as TF-IDF vectors and ranks candidates by cosine similarity to a movie the user already likes.
- **Collaborative Filtering** — applies Truncated SVD to the user-item ratings matrix to estimate how a specific user would rate unseen movies, capturing patterns from the wider community of users.

A **Hybrid Engine** merges both signals using a configurable weight, so the user can shift the balance between *"movies similar to this one"* and *"movies people like me enjoyed."*

The system is built on the **MovieLens** dataset (`movies.csv`, `ratings.csv`) and exposed through a Netflix-styled Streamlit dashboard.

---

## ✨ Features

- 🔎 **Movie-based search** — pick any watched movie as the recommendation seed.
- 🎚️ **Adjustable hybrid weight** — a live slider blends content-based and collaborative scores.
- 👤 **Per-user personalization** — recommendations are computed for a specific `userId`.
- 📊 **Live performance metrics** — RMSE is calculated on-the-fly from a sampled test set; Precision@K and latency are displayed alongside it.
- 🎨 **Custom dark UI** — Netflix-inspired theme built with Streamlit's CSS injection.

---

## 🗂️ Project Structure

```
MovieStream-Ai/
├── app.py                      # Streamlit application entry point
├── requirements.txt            # Python dependencies
├── data/
│   ├── movies.csv              # Raw MovieLens movies metadata
│   ├── ratings.csv             # Raw MovieLens user ratings
│   ├── processed_movies.csv    # Cleaned movies + engineered features
│   └── user_item_matrix.pkl    # SVD-reduced user-item matrix
├── ml_model/
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer (genres)
│   └── svd_model.pkl           # Fitted TruncatedSVD model
├── notebooks/
│   └── eda.ipynb               # Exploratory data analysis
└── src/
    ├── preprocessing.py        # Data cleaning + TF-IDF fitting
    ├── content_model.py        # Content-based recommender
    ├── collaborative_model.py  # Collaborative (SVD) recommender
    ├── hybrid_engine.py        # Combines both models into final scores
    └── train_svd.py            # Trains and persists the SVD model
```

---

## ⚙️ How It Works

### 1. Preprocessing (`src/preprocessing.py`)
- Loads raw `movies.csv` and `ratings.csv`.
- Removes duplicate/missing entries and strips release years from titles (e.g. `"Toy Story (1995)"` → `"Toy Story"`).
- Converts the `genres` column (pipe-separated) into space-separated text and fits a `TfidfVectorizer` on it.
- Joins average rating and rating count per movie from the ratings data.
- Persists the vectorizer to `ml_model/tfidf_vectorizer.pkl` and the enriched dataset to `data/processed_movies.csv`.

### 2. Collaborative Model Training (`src/train_svd.py`)
- Pivots `ratings.csv` into a user × movie matrix.
- Applies `TruncatedSVD` (50 latent components) on its transpose to obtain a reduced movie-latent representation.
- Saves the fitted model (`svd_model.pkl`) and the reduced matrix (`user_item_matrix.pkl`).

### 3. Content-Based Recommender (`src/content_model.py`)
- Computes cosine similarity between a selected movie's TF-IDF vector and all other movies' vectors.
- Returns the top-N most genre-similar titles.

### 4. Collaborative Recommender (`src/collaborative_model.py`)
- Predicts a rating for a given `(user_id, movie_id)` pair using the SVD-reduced latent matrix.
- Includes an RMSE evaluator that compares predictions against a sample of real ratings.

### 5. Hybrid Engine (`src/hybrid_engine.py`)
- Pulls the top-20 content-based candidates for the selected movie.
- Scores each candidate as:

  ```
  final_score = (content_weight × content_score) + ((1 − content_weight) × normalized_predicted_rating)
  ```

- Sorts and returns the top-N recommendations for the requesting user.

### 6. Streamlit App (`app.py`)
- Lets the user pick a movie, a `userId`, the number of recommendations, and the content/collaborative weight.
- Displays results as styled cards with their hybrid score.
- Shows live system metrics (RMSE, Precision@K, latency) in an expandable panel.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Mahmoud-Mansour5/MovieStream-Ai.git
cd MovieStream-Ai
pip install -r requirements.txt
```

### (Re)build the models — optional

The repository already ships with trained artifacts in `ml_model/` and `data/`. To regenerate them from scratch:

```bash
# 1. Preprocess raw data + fit the TF-IDF vectorizer
python src/preprocessing.py

# 2. Train the SVD collaborative model
python src/train_svd.py
```

### Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

---

## 🧪 Usage

1. Open the app in your browser.
2. In the sidebar, enter a **User ID** (1–610) and adjust the **Content-Based Weight** slider.
3. Select a movie you've watched from the dropdown.
4. Choose how many recommendations you want (5/10/15/20).
5. Click **Generate Recommendations** to see the hybrid results, each with its computed score.
6. Expand **View System Performance Metrics** to inspect RMSE, Precision@K, and latency.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI / App | Streamlit |
| Data manipulation | pandas, NumPy |
| ML — content-based | scikit-learn (`TfidfVectorizer`, cosine similarity) |
| ML — collaborative | scikit-learn (`TruncatedSVD`) |
| Model persistence | joblib |
| Dataset | MovieLens (`movies.csv`, `ratings.csv`) |

---

## 📈 Dataset

This project uses the **MovieLens** dataset:
- `movies.csv` — movie IDs, titles, and pipe-separated genres.
- `ratings.csv` — user ratings (`userId`, `movieId`, `rating`, `timestamp`).

---

## 🗺️ Roadmap

- [ ] Replace the heuristic SVD-based rating predictor with a true reconstructed user-item prediction (`user_vector · item_vector`).
- [ ] Add collaborative filtering evaluation metrics (Precision@K / Recall@K) computed from real data instead of static placeholders.
- [ ] Cache TF-IDF similarity matrix to avoid recomputation per request.
- [ ] Add automated tests for `content_model`, `collaborative_model`, and `hybrid_engine`.
- [ ] Containerize with Docker for one-command deployment.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Mahmoud Mansour**
GitHub: [@Mahmoud-Mansour5](https://github.com/Mahmoud-Mansour5)
