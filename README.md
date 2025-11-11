# 🎬 Animation Recommendation System

## 📊 Data Inspection

Both **`anime.csv`** and **`rating_complete.csv`** datasets are clean, consistent, and well-structured.  
There are no missing or unrated values, and the data distribution reflects typical user behavior —  
a strong preference for popular titles and high average ratings.

These characteristics make the dataset well-suited for developing a **hybrid recommendation system**  
that combines **Collaborative Filtering (CF)** with **Content-Based (CB)** embeddings.
<img width="940" height="726" alt="image" src="https://github.com/user-attachments/assets/6f115562-c96c-4866-8133-c03a32525b14" />
<img width="535" height="1251" alt="image" src="https://github.com/user-attachments/assets/bca95216-f8af-4f3d-a975-59082fc68691" />
<img width="929" height="1151" alt="image" src="https://github.com/user-attachments/assets/6cc81d83-16c5-48c4-b738-ede3c80df971" />
<img width="563" height="1239" alt="image" src="https://github.com/user-attachments/assets/c4f8cdeb-86b2-4d87-9c42-ef71cef08f34" />
<img width="940" height="989" alt="image" src="https://github.com/user-attachments/assets/a5fa5adb-82c7-40c6-adef-7f6b2b47d5fd" />

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


## 📦 Modeling & Evaluation

---

## 1. Data and Overall Pipeline

### 1.1 Input Data

- `rating_complete.csv`
  - Columns: `user_id`, `anime_id`, `rating`
  - Only ratings > 0 are used.

- `rating_test.csv`
  - Used for baseline CF models (e.g., SVD, NeuMF).

- `meta_preprocessed.csv`
  - Item-level content features (genres, tags, staff, etc.) + `MAL_ID`.
  - Output of the preprocessing pipeline.

- `anime.csv`
  - Mapping table for `MAL_ID ↔ anime title` (for human-readable outputs).

### 1.2 Pipeline Overview

1. **Collaborative Filtering (CF) – SVD**
   - Learns latent user/item factors.
   - Outputs: `cf_score(user_id, anime_id)`.

2. **Content-Based (CB) – Cosine Similarity**
   - Uses standardized content embeddings from `meta_preprocessed.csv`.
   - Builds per-user content profiles.
   - Outputs: `cb_score(user_id, anime_id)`.

3. **Meta-Training Dataset**
   - For sampled user–item pairs:
     - Collect `(cf_score, cb_score, true_rating)`.
   - Used as input for meta-learners.

4. **Meta-Learners (XGBoost / LightGBM / CatBoost)**
   - Learn a mapping:
     - $\hat{y} = f_{\text{meta}}(y_{\text{cf}}, y_{\text{cb}})$
   - Final hybrid rating predictor.

5. **Evaluation Metrics**
   - **RMSE (raw)**: absolute prediction error.
   - **Precision@K / Recall@K**: ranking-based accuracy.
   - **Model-based item recommendation**:
     - “Users who liked X also tend to like Y” based on meta-model predictions.

---

## 2. Collaborative Filtering (SVD)

### 2.1 Model Description

- Library: `surprise` (SVD)
- Predictive model:
  
  $\hat{r}_{ui} = \mu + b_u + b_i + p_u^\top q_i$
  
  - $\mu$: global mean
  - $b_u$, $b_i$: user/item bias
  - $p_u$, $q_i$: latent factor vectors.

### 2.2 Key Hyperparameters

| Parameter   | Value  | Meaning              | Effect                                   |
|------------|--------|----------------------|------------------------------------------|
| `n_factors`| 100    | Latent dimension     | Higher → expressive, risk of overfitting |
| `n_epochs` | 20     | Training epochs      | Ensures convergence                      |
| `lr_all`   | 0.005  | Learning rate        | Stable, small updates                    |
| `reg_all`  | 0.02   | Regularization       | Prevents overfitting                     |

- Output:
  - `cf_score(u, i) = svd.predict(u, i).est`

---

### (Additional) Collaborative Filtering Using NeuMF

**Concept**

- NeuMF embeds users/items via:
  1. **GMF** (Generalized Matrix Factorization): linear interaction.
  2. **MLP**: non-linear interaction.
- Embeddings:
  - User: $p_u$
  - Item: $q_i$

Key components:

- **GMF**:
  - $\phi_{\text{GMF}}(p_u, q_i) = p_u \odot q_i$
- **MLP**:
  - Stacked non-linear layers on concatenated $(p_u, q_i)$.
- **Fusion**:
  - Final prediction from concatenation of GMF + MLP representations.

**Analysis**

- NeuMF can capture complex non-linear patterns that SVD cannot.
- However, in our setting:
  - Rating skew: ~70% in 8–9 range.
  - Model size: ~33K parameters.
  - Resulted in **overfitting**:
    - Train RMSE: **1.27**
    - Val RMSE: **1.80**
- Final decision:
  - **SVD selected** as primary CF model:
    - Stable performance
    - Validation RMSE: **1.70**

---

## 3. Content-Based (CB) Module

### 3.1 Embedding Preprocessing

- Uses `meta_preprocessed.csv`.
- Applies `StandardScaler` to CB features.
- Ensures cosine similarity is not dominated by large-scale features.

### 3.2 Item–Item Cosine Similarity

- For items $i, j$ with vectors $x_i, x_j$:

  $\text{sim}(i, j) = \dfrac{x_i \cdot x_j}{\|x_i\| \, \|x_j\|}$

- Similarity matrix shape: `(n_items × n_items)`.

### 3.3 User Content Profile (Top-K Neighbor Masking)

For each user $u$:

1. Define liked items:
   - $L_u = \{ i \mid r_{ui} \ge \tau \}$, with default $\tau = 8.0$.
2. For each liked item $i \in L_u$:
   - Keep only Top-K most similar neighbors $S_i$.
3. User-specific CB score for candidate item $j$:

   $$
   \text{CB}_u(j)
   = \frac{\sum_{i \in L_u} \mathbf{1}\{ j \in \text{TopK}(S_i) \} \cdot \text{sim}(i, j)}
          {\sum_{i \in L_u} \mathbf{1}\{ j \in \text{TopK}(S_i) \} + \varepsilon}
   $$

- Default:
  - `top_k = 30`
  - `like_th = 8.0`

**Rationale**

- Focuses on **strongest semantic neighbors**.
- Reduces noise from weakly related items.

---

## 4. Meta-Training Dataset Construction

### 4.1 User Sampling

- Randomly sample up to **3000 users** for efficiency.
- For each sampled user:
  - Build CB profile.
  - For each rated item $(u, i)$:
    - Compute `cf_score(u, i)` from SVD.
    - Compute `cb_score(u, i)` from CB module.
    - Take `true_rating(u, i)` from `rating_complete.csv`.
    - Store: `(user_id, anime_id, cf_score, cb_score, true_rating)`.

### 4.2 Output

- Save as: `meta_train_ready.csv`.
- Used as supervised training data for meta-learners.

---

## 5. Meta-Learners (XGBoost / LightGBM / CatBoost)

### 5.1 Concept of the Meta-Learner

The meta-learner is a **fusion model**:

- Input:
  - $y_{\text{cf}}$: CF-predicted rating
  - $y_{\text{cb}}$: CB-predicted rating
- Output:
  - $\hat{y} = f_{\text{meta}}(y_{\text{cf}}, y_{\text{cb}})$

Objective:

$$
\min_{f_{\text{meta}}} \mathbb{E}\left[ \left( y - f_{\text{meta}}(y_{\text{cf}}, y_{\text{cb}}) \right)^2 \right]
$$

- Learns how to combine:
  - Behavioral signal (CF)
  - Semantic signal (CB)

---

### 5.2 System Implementation Process

1. **Base Model Training**
   - CF (SVD): trained on `rating_complete.csv`.
   - CB: similarity + user content profiles from `meta_preprocessed.csv`.

2. **Meta-Dataset Construction**
   - For each $(u, i)`:
     - Features: `[y_{\text{cf}}(u, i), y_{\text{cb}}(u, i)]`
     - Target: `true_rating(u, i)`
   - Stored in `meta_train_ready.csv`.

3. **Meta-Learner Training**
   - Train **LightGBM / XGBoost / CatBoost**:
     - $\hat{y}_{ui} = f_{\text{meta}}(y_{\text{cf}}, y_{\text{cb}})$

4. **Recommendation Generation**
   - For each user:
     - Predict $\hat{y}_{ui}$ for candidate items.
     - Sort descending → Top-N recommendations.

---

### 5.3 Characteristics of Each Meta-Learner

**(1) LightGBM**

- Leaf-wise, histogram-based Gradient Boosting.
- Key:
  - Fast convergence
  - Efficient memory usage
  - GOSS sampling
- Chosen as **final meta-learner** in this project:
  - Best trade-off between speed, performance, and interpretability.

**(2) XGBoost**

- Level-wise tree growth.
- L1/L2 regularization.
- Robust, stable, handles missing values well.
- Slightly slower but very reliable.

**(3) CatBoost**

- Ordered boosting + Oblivious Trees.
- Great with categorical features & overfitting control.
- In our case (continuous inputs), still stable but less beneficial than LightGBM.

---

### 5.4 Model Comparison Summary

| Model     | Strengths                                  | Limitations                   |
|----------|---------------------------------------------|-------------------------------|
| LightGBM | Fast, accurate, low memory, interpretable   | Leaf-wise can overfit         |
| XGBoost  | Stable, strong regularization, robust       | Slower than LightGBM          |
| CatBoost | Great with categoricals, robust to overfit  | Slower on large-scale tasks   |

---

### 5.5 Functional Interpretation

The hybrid prediction can be viewed as:

$$
\hat{y}_{ui} \approx w_1 \cdot y_{\text{cf}} + w_2 \cdot y_{\text{cb}} + b
$$

- But:
  - In Gradient Boosting Trees, $w_1, w_2$ are **implicit, non-linear, context-dependent**.
  - The meta-learner:
    - Reinforces agreement between CF & CB.
    - Downweights the less reliable model per user/genre pattern.

**Conclusion**

- The meta-learner acts as an **adaptive fusion layer**.
- Among candidates, **LightGBM** showed the best overall performance and was selected as the final meta-model.

---

## 6. Evaluation

### 6.1 RMSE (Raw Predictions)

$$
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{(u,i)} (\hat{r}_{ui} - r_{ui})^2 }
$$

- Computed on raw predictions (no scaling/clipping).
- Lower RMSE → better numeric accuracy.

### 6.2 Precision@K / Recall@K

- Relevant items:
  - $r_{ui} \ge 8.0$
- Metrics:

  $$
  \text{Precision@K} = \frac{|T_u \cap R_u|}{K}, \quad
  \text{Recall@K} = \frac{|T_u \cap R_u|}{|R_u|}
  $$

  - $T_u$: Top-K recommended items.
  - $R_u$: Relevant (truly liked) items.

**Why use raw predictions**

- Rank-based metrics are invariant to monotonic transforms.
- No clipping/scaling before ranking → preserves relative order.

---

## 7. Sample User Top-10 Comparison

- Select a random user.
- Generate Top-10 lists using:
  - XGBoost-based meta-learner
  - LightGBM-based meta-learner
  - CatBoost-based meta-learner
- Display:
  - `Predicted Rating / True Rating` per item.
- Provides qualitative insight into each model’s personalization behavior.

---

## 8. Model-Based “Similar Item” Recommendation

### 8.1 Concept

- Recommend items that users who liked a given anime also tend to like.
- Uses meta-model predictions instead of pure co-occurrence:
  - Higher semantic + behavioral resolution.

### 8.2 Steps

1. Find target anime by name → get `MAL_ID`.
2. Find users who rated that anime ≥ 8.0.
3. Collect other items those users rated.
4. For each `(user, item)`:
   - Gather `cf_score`, `cb_score` (from meta_train_ready or recomputation).
   - Predict via meta-model.
5. Average predicted scores per item.
6. Sort in descending order → Top-N similar items.

- **Clipping**:
  - Predictions may be clipped to `[1, 10]` **for display only**.
  - Training & evaluation use raw values.

Example interpretation:
- “Items that users who liked **Koe no Katachi** are also highly likely to enjoy.”

---

## 9. Design Choices & Rationale

- **Top-K neighbor masking**
  - Reduces noise; focuses on strongest neighbors.

- **Raw predictions (no global scaling)**
  - Each model learns the rating scale directly.
  - Avoids distortions in ranking and RMSE.

- **User sampling (3000 users)**
  - Practical runtime with sufficient diversity.
  - For production:
    - Use distributed compute or ANN libraries (e.g., FAISS).

---

## 10. Computational Complexity and Execution Notes

| Component             | Complexity          | Comment                                             |
|----------------------|---------------------|-----------------------------------------------------|
| Item similarity      | $O(N^2 d)$          | Expensive; ANN/FAISS/Annoy recommended at scale    |
| Meta dataset build   | $O(U \cdot \bar{I})$| $U$: users, $\bar{I}$: avg rated items             |
| Model training       | Linear in samples   | Only 2 features (cf, cb) → very fast               |
| Reproducibility      | —                   | `random_state=42`, `np.random.seed(42)` used       |

---

## 11. Result Interpretation Guidelines

- **RMSE**
  - Lower = better absolute predictive accuracy.

- **Precision@10 / Recall@10**
  - Reflect ranking quality of recommendations.

- **User Top-10 lists**
  - Reveal each model’s tendency:
    - Safe vs. exploratory
    - Niche vs. popular bias.

### Result of NeuMF
<img width="658" height="360" alt="image" src="https://github.com/user-attachments/assets/3d91dd20-7ee1-4eda-b570-4b2598618ab5" />

  ### Result of Hybrid model
<img width="716" height="313" alt="image" src="https://github.com/user-attachments/assets/049a2d69-bb98-404f-8087-cc2723a65b24" />
<img width="940" height="234" alt="image" src="https://github.com/user-attachments/assets/7e795ad2-6161-4a22-a147-20e7c20bcba3" />
<img width="940" height="946" alt="image" src="https://github.com/user-attachments/assets/b3bca91a-7642-4e39-a019-3f9269ab1d39" />
  
  ### Only SVD
<img width="569" height="729" alt="image" src="https://github.com/user-attachments/assets/991ecd31-f687-43f7-9066-d1e5312e2d8c" />

---

## Visualization of anime embeddings
<img width="940" height="310" alt="image" src="https://github.com/user-attachments/assets/ea1f7f4b-e534-4224-beeb-ec114a9596df" />
<img width="940" height="353" alt="image" src="https://github.com/user-attachments/assets/ee47765b-bb81-4c4a-8e42-bed66b79be7c" />

We can observe that anime belonging to the same genre tend to form clusters in the embedding space.
---

## Visualizing Hybrid Meta-Learner Recommendations
<img width="940" height="361" alt="image" src="https://github.com/user-attachments/assets/9224e195-2ca3-47a0-b722-fe761c92ab19" />
<img width="940" height="373" alt="image" src="https://github.com/user-attachments/assets/f45bcf3b-0630-49c4-bd73-e9e2951b116a" />
<img width="940" height="341" alt="image" src="https://github.com/user-attachments/assets/a3f42c02-e62d-4e18-9206-f84cffff41f3" />
<img width="940" height="306" alt="image" src="https://github.com/user-attachments/assets/a5e7712c-fb33-4816-aff3-9c7bd321e8ac" />

We can see that the model recommended an animation with "action" and "fantasy," which are the genre that the user likes. Here, the model used the light GBM, which has best performance. We can also see that the animation recommended by the model shows that the same genre is clustered.

---

## Recommendation System GUI

<img width="940" height="635" alt="image" src="https://github.com/user-attachments/assets/36a4bba0-8eb5-4e00-a864-6ca35146c34e" />
<img width="940" height="632" alt="image" src="https://github.com/user-attachments/assets/c5cc091c-572e-4636-b795-a4916b9e9b86" />


---

### 🧾 Conclusion

The **NeuMF** model achieved a **lower RMSE** than the **SVD** model.

Among the three meta-learner models (**XGBoost**, **LightGBM**, and **CatBoost**),  
**LightGBM** achieved the **best overall performance** in both **RMSE** and **ranking metrics** (Precision@K, Recall@K).

This can be attributed to several key factors:

1. **Histogram-based Gradient Boosting Efficiency**  
   LightGBM groups continuous feature values into discrete bins (histogram-based splitting).  
   This approach enables **faster training**, **smoother optimization**, and **noise reduction**,  
   particularly with small meta-feature dimensions such as `(cf_score, cb_score)`.

2. **Better Handling of Continuous Features**  
   Since the meta-input consists of two continuous scores (from CF and CB),  
   LightGBM’s **leaf-wise growth** captures **subtle nonlinear interactions** more effectively  
   than XGBoost’s level-wise tree expansion.

3. **Regularization and Feature Usage**  
   LightGBM automatically prunes weak leaves during training, providing natural **regularization**.  
   This results in **better generalization** across users with diverse rating behaviors.

4. **Optimized for Small, Dense Features**  
   While CatBoost excels with high-cardinality categorical data,  
   LightGBM performs **exceptionally well on low-dimensional numeric features**,  
   avoiding unnecessary complexity.

**Summary:**  
Among all ensemble meta-learners, **LightGBM** demonstrated the **highest predictive accuracy**.  
Its histogram-based optimization and efficient handling of low-dimensional continuous features  
provided smoother gradient updates, reduced overfitting, and yielded superior generalization performance.

