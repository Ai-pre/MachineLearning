# ============================================
# 0. Data Inspection (anime.csv / rating_complete.csv)
# ============================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Datasets
# -----------------------------
ANIME_PATH = "anime.csv"
RATING_PATH = "rating_complete.csv"
anime_df = pd.read_csv(ANIME_PATH)
rating_df = pd.read_csv(RATING_PATH)
print(f"Loaded anime.csv: {anime_df.shape}")
print(f"Loaded rating_complete.csv: {rating_df.shape}")

# -----------------------------
# 2. Basic Overview
# -----------------------------
print("\n====================================")
print("Anime.csv Overview")
print("====================================")
print("\nAnime Columns:")
print(anime_df.columns.tolist())
print("\nAnime Data Sample:")
print(anime_df.head())
print("\nAnime Info:")
print(anime_df.info())

# -----------------------------
# 3. Missing Value Check
# -----------------------------
missing_ratio = anime_df.isnull().mean().sort_values(ascending=False)
if (missing_ratio == 0).all():
    print("\nNo missing values found in anime.csv.")
else:
    print("\nMissing Values (Top 10):")
    print(missing_ratio.head(10))
    
# -----------------------------
# 4. Anime Statistics
# -----------------------------
print("\nNumeric Summary:")
print(anime_df.describe())
print("\nAnime Type Distribution:")
print(anime_df['Type'].value_counts())
print("\nRating Category Distribution:")
print(anime_df['Rating'].value_counts())

# -----------------------------
# 5. Rating Dataset Overview
# -----------------------------
print("\n====================================")
print("rating_complete.csv Overview")
print("====================================")
print("\nRating Columns:")
print(rating_df.columns.tolist())
print("\nRating Data Sample:")
print(rating_df.head())
print("\nRating Info:")
print(rating_df.info())
# Unique users and items
num_users = rating_df['user_id'].nunique()
num_anime = rating_df['anime_id'].nunique()
print(f"\nUnique Users: {num_users}")
print(f"Unique Anime: {num_anime}")
# Rating distribution
print("\nRating Distribution:")
print(rating_df['rating'].value_counts().sort_index())
# Count rating == 0 (unrated entries)
zero_count = (rating_df['rating'] == 0).sum()
print(f"\nUnrated (rating=0): {zero_count} / {len(rating_df)}")

# -----------------------------
# 6. Optional Visualizations
# -----------------------------
# Rating distribution
plt.figure(figsize=(6,4))
sns.countplot(x='rating', data=rating_df, color='steelblue')
plt.title("Rating Distribution (rating_complete.csv)")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()
# Anime type distribution
plt.figure(figsize=(8,4))
sns.countplot(y='Type', data=anime_df, order=anime_df['Type'].value_counts().index, color='coral')
plt.title("Anime Type Distribution")
plt.show()

# ============================================
#   meta_preprocessed.csv Builder
#   → Content-Based Features for Hybrid Recommender
#   Includes: TF-IDF (Genres/Producers/Studios),
#             Label Encoders for categorical fields,
#             Scaled numeric features (Score, Popularity, etc.)
# ============================================

import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------------------------------------
# 1️) Load anime metadata and handle missing values
# ------------------------------------------------------------
"""
- anime.csv includes all raw item metadata (Genres, Studios, Type, etc.)
- Missing categorical or text fields must be replaced with empty strings
  to ensure vectorizers and encoders do not fail.
- MAL_ID (unique anime identifier) is normalized to integer.
"""
meta_df = pd.read_csv("anime.csv")
print(f"Loaded anime.csv, shape: {meta_df.shape}")

# Fill missing fields with default values
meta_df = meta_df.fillna({
    'Genres': '',
    'Producers': '',
    'Studios': '',
    'Licensors': '',
    'Type': 'Unknown',
    'Source': 'Unknown',
    'Rating': 'Unknown',
    'Premiered': 'Unknown',
    'Duration': 'Unknown'
})

# Normalize ID type
meta_df['MAL_ID'] = meta_df['MAL_ID'].astype(int)

# ------------------------------------------------------------
# 2️) Encode categorical features (for numeric representation)
# ------------------------------------------------------------
"""
- LabelEncoder converts string labels into integer codes.
- Encoders are saved in a dictionary for later decoding or reuse.
- Target columns: Type, Source, Rating, Premiered, Duration
"""
label_cols = ['Type', 'Source', 'Rating', 'Premiered', 'Duration']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    meta_df[f'{col}_encoded'] = le.fit_transform(meta_df[col].astype(str))
    label_encoders[col] = le

# ------------------------------------------------------------
# 3️) Create TF-IDF features for multi-valued string columns
# ------------------------------------------------------------
"""
- Genres, Producers, and Studios contain comma-separated tokens.
- TF-IDF vectorization is used to represent each anime as a sparse semantic vector.
- token_pattern ensures splitting by commas or spaces.
- max_features=100 limits feature dimensionality for computational efficiency.
"""
def make_tfidf_features(df, column, prefix):
    tfidf = TfidfVectorizer(
        token_pattern=r'[^, ]+',     # split by comma or space
        stop_words='english',
        max_features=100
    )
    mat = tfidf.fit_transform(df[column])
    tfidf_df = pd.DataFrame(
        mat.toarray(),
        columns=[f"{prefix}_{t}" for t in tfidf.get_feature_names_out()]
    )
    return tfidf_df, tfidf

# Apply TF-IDF to selected columns
tfidf_genres, vec_genres = make_tfidf_features(meta_df, 'Genres', 'Genre')
tfidf_producers, vec_producers = make_tfidf_features(meta_df, 'Producers', 'Prod')
tfidf_studios, vec_studios = make_tfidf_features(meta_df, 'Studios', 'Studio')

# ------------------------------------------------------------
# 4️) Handle numeric features (normalize for scale balance)
# ------------------------------------------------------------
"""
- Convert numeric fields to float type (coercing invalid values to NaN).
- Fill missing numeric values with column means to prevent bias.
- Apply MinMaxScaler (0–1 normalization) for consistent scale across features.
"""
numeric_cols = ['Score', 'Episodes', 'Ranked', 'Popularity', 'Members', 'Favorites']
for col in numeric_cols:
    meta_df[col] = pd.to_numeric(meta_df[col], errors='coerce')
meta_df[numeric_cols] = meta_df[numeric_cols].fillna(meta_df[numeric_cols].mean())

scaler = MinMaxScaler()
meta_df[numeric_cols] = scaler.fit_transform(meta_df[numeric_cols])

# ------------------------------------------------------------
# 5️) Combine all Content-Based (CB) features
# ------------------------------------------------------------
"""
- Combine:
  (a) Normalized numeric columns
  (b) Encoded categorical columns
  (c) TF-IDF features (Genres, Producers, Studios)
- Result: dense vector representation for each anime
"""
meta_processed = pd.concat(
    [
        meta_df[['MAL_ID'] + numeric_cols + [f'{col}_encoded' for col in label_cols]],
        tfidf_genres,
        tfidf_producers,
        tfidf_studios
    ],
    axis=1
)

# ------------------------------------------------------------
# 6️) Match anime IDs with rating_complete.csv
# ------------------------------------------------------------
"""
- rating_complete.csv contains actual user–anime interactions.
- Filter out anime with no ratings, keeping only valid MAL_IDs.
- Ensures that every content vector aligns with CF-based entries later.
"""
rating_df = pd.read_csv("rating_complete.csv")
rating_df['anime_id'] = rating_df['anime_id'].astype(int)

valid_ids = set(rating_df['anime_id']).intersection(set(meta_processed['MAL_ID']))
meta_processed = meta_processed[meta_processed['MAL_ID'].isin(valid_ids)].reset_index(drop=True)

print(f"Matched items with rating data: {len(valid_ids)} / {len(meta_df)}")
print(f"Final processed shape: {meta_processed.shape}")

# ------------------------------------------------------------
# 7️) Save the processed data and vectorizers
# ------------------------------------------------------------
"""
- Save meta_preprocessed.csv for downstream hybrid model training.
- Optional: save TF-IDF vectorizers, encoders, and scaler via pickle
  for reproducibility (not required unless reusing transformations).
"""
meta_processed.to_csv("meta_preprocessed.csv", index=False)
print("Saved: meta_preprocessed.csv")

# Optionally save fitted preprocessing objects
with open("tfidf_and_encoders.pkl", "wb") as f:
    pickle.dump({
        "label_encoders": label_encoders,
        "vec_genres": vec_genres,
        "vec_producers": vec_producers,
        "vec_studios": vec_studios,
        "scaler": scaler
    }, f)

print("Saved: tfidf_and_encoders.pkl (vectorizers + encoders + scaler)")
