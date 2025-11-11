# 🎬 Animation Recommendation System

## 📊 Data Inspection

Both **`anime.csv`** and **`rating_complete.csv`** datasets are clean, consistent, and well-structured.  
There are no missing or unrated values, and the data distribution reflects typical user behavior —  
a strong preference for popular titles and high average ratings.

These characteristics make the dataset well-suited for developing a **hybrid recommendation system**  
that combines **Collaborative Filtering (CF)** with **Content-Based (CB)** embeddings.

---

## 🧼 Preprocessing Pipeline

### 1️⃣ Load & Clean Raw Metadata

**What it does**
- Reads `anime.csv` and normalizes the identifier `MAL_ID` to `int`.
- Fills missing values:
  - Text/multi-valued fields → `""` (empty string, so TF-IDF won’t crash)
  - Categorical fields (`Type`, `Source`, `Rating`, `Premiered`, `Duration`) → `"Unknown"`  
- Ensures all downstream transformers receive valid inputs.

**Why it matters**
- Vectorizers/encoders don’t accept `NaN`.
- Consistent typing on IDs prevents silent mismatches in later joins.

**Pitfalls & mitigations**
- Ambiguous categories: using `"Unknown"` groups all missing categories together — fine for modeling, but call it out in limitations.
- Unicode & punctuation: TF-IDF tokenization later uses a strict pattern; commas/spaces separate tokens.

**Quick checks**
```python
meta_df.isna().sum()
meta_df['MAL_ID'].dtype
```

### 2️⃣ Encode Categorical Features

**What it does**
- Applies `LabelEncoder` to: `Type`, `Source`, `Rating`, `Premiered`, `Duration`.
- Stores each fitted encoder in `label_encoders` for reproducibility/inference.

**Why it matters**
- Tree models and similarity pipelines consume numeric inputs.
- Persisted encoders allow consistent mapping when new/held-out items arrive.

**Pitfalls & mitigations**
- Unseen categories at inference: `LabelEncoder` will error if a new label appears.  
  - Option 1️⃣: Pre-map unknowns to a reserved index.  
  - Option 2️⃣: Refit encoders on union of train + incoming batch (document this policy).

**Quick checks**
```python
meta_df.filter(like='_encoded').nunique()
```

### 3️⃣ Build TF-IDF Features (Genres / Producers / Studios)

**What it does**
- Runs **TF-IDF** with:
  - `token_pattern=r'[^, ]+'` (split by commas/spaces)
  - `stop_words='english'`
  - `max_features=100` per field.
- Produces three dense matrices with interpretable columns like:
  - `Genre_Action`
  - `Prod_Aniplex`
  - `Studio_KyoAni`

**Why it matters**
- Converts multi-label text fields into semantic vectors.
- Capped dimensionality reduces memory usage and speeds up cosine similarity later.

**Pitfalls & mitigations**
- **Vocabulary drift:**  
  Save the vectorizers for inference (`tfidf_and_encoders.pkl`) to ensure consistent feature space.
- **Over-sparsity:**  
  Using `max_features=100` balances expressiveness and compute efficiency.

**Quick checks**
```python
tfidf_genres.shape[1] == 100
vec_genres.get_feature_names_out()[:10]
```

### 4️⃣ Normalize Numeric Attributes

**What it does**
- Casts numeric columns:  
  `['Score', 'Episodes', 'Ranked', 'Popularity', 'Members', 'Favorites']`  
  to numeric (`errors='coerce'`).
- Fills remaining `NaN`s with column means.
- Applies `MinMaxScaler` to range `[0, 1]`.

**Why it matters**
- Mixed-scale features (e.g., `Members` in millions vs. `Score` ~ [0–10])  
  can distort distance metrics and degrade model performance.
- Normalization stabilizes training and improves cosine similarity reliability.

**Pitfalls & mitigations**
- **Outliers:**  
  `MinMaxScaler` is sensitive — if the distribution is heavy-tailed  
  (e.g., `Members`), consider using `RobustScaler` in future.
- **Imputation bias:**  
  Mean imputation is simple but can bias toward central values;  
  mention this in your limitations section.

**Quick checks**
```python
meta_df[numeric_cols].min().ge(0).all()
meta_df[numeric_cols].max().le(1).all()
```

### 5️⃣ Assemble the Content-Based Feature Table

**What it does**
- Concatenates:
  1. `MAL_ID`
  2. Scaled numerics
  3. Encoded categoricals
  4. TF-IDF matrices (`Genres`, `Producers`, `Studios`)
- Produces a **dense, model-ready Content-Based (CB) vector** per anime.

**Why it matters**
- Combines all preprocessed data sources into a single, unified representation.
- This unified table can be directly used for:
  - Cosine similarity computation between items
  - Feature input for the hybrid meta-learner

**Pitfalls & mitigations**
- **Column alignment:**  
  Ensure consistent row alignment when concatenating multiple feature sets.  
  `pd.concat(..., axis=1)` relies on aligned indices.
- **Shape mismatch:**  
  Verify that TF-IDF outputs and encoded numerics have identical row counts.

**Quick checks**
```python
meta_processed.shape[0] == meta_df.shape[0]
```

### 6️⃣ Align with Interaction Data (`rating_complete.csv`)

**What it does**
- Reads `rating_complete.csv` and casts `anime_id` to `int`.
- Intersects `MAL_ID` from the metadata with `anime_id` from the ratings file.
- Keeps only the overlapping records — ensures every CB vector has a corresponding CF entry.
- Logs the number of matched items for transparency.

**Why it matters**
- Prevents **cold-start leakage** during hybrid model training and evaluation.
- Guarantees 1:1 correspondence between items in the CB feature matrix  
  and those in the CF (Collaborative Filtering) dataset.

**Pitfalls & mitigations**
- **Coverage drop:**  
  Titles without rating data will be dropped from the hybrid dataset.  
  It’s recommended to log the retained ratio for your report (e.g., `matched / total`).
- **ID mismatch:**  
  Ensure both sides (`MAL_ID`, `anime_id`) are of the same integer type  
  to avoid silent filtering errors.

**Quick checks**
```python
(meta_processed['MAL_ID'].isin(rating_df['anime_id'])).all()
```

### 7️⃣ Persist Outputs for Reproducibility

**What it does**
- Saves:
  - `meta_preprocessed.csv` → the final, dense CB feature table
  - `tfidf_and_encoders.pkl` → serialized preprocessing objects containing:
    - `label_encoders` (for categorical columns)
    - `vec_genres`, `vec_producers`, `vec_studios`
    - `scaler`

**Why it matters**
- Ensures **identical preprocessing** during both training and inference.
- Prevents **train–serve feature drift** by preserving:
  - TF-IDF vocabularies
  - LabelEncoder mappings
  - Scaler parameters

**Pitfalls & mitigations**
- Always reload the pickle and test `transform()` on a sample before deployment  
  to confirm version compatibility and consistent output dimensions.

**Quick checks**
```python
import pickle
obj = pickle.load(open("tfidf_and_encoders.pkl", "rb"))
type(obj)
```



