# ======================================================
# CF(SVD) + CB(Content-Based) Hybrid Meta-Train Builder
#     + XGBoost vs LightGBM vs CatBoost (raw predictions)
#     + Model-based "similar items" recommender
#     * Only ranking & display use clipping to [1, 10]
# ======================================================

"""
Overview
--------
This script builds a hybrid recommendation pipeline:

1) CF (Collaborative Filtering) with Surprise SVD:
   - Learns latent user/item factors from explicit ratings.

2) CB (Content-Based) similarity over TF-IDF-like item features:
   - Computes an item-item cosine similarity matrix over preprocessed content vectors.
   - For each user, aggregates only the Top-K most similar neighbors of the user's liked items.
   - Produces a per-user "content profile" to score candidate items (CB score).

3) Meta-Training Dataset:
   - For sampled users, pair (cf_score, cb_score, true_rating) per (user, item).

4) Meta-Learners:
   - XGBoost, LightGBM, CatBoost regressors trained to predict true ratings from (cf_score, cb_score).
   - RMSE evaluation uses raw predictions (no clipping).

5) Ranking Metrics:
   - Precision@K, Recall@K use predictions clipped to [1, 10] to avoid ranking distortion by overshoot.
   - A rating ≥ 8.0 is considered "relevant".

6) Model-based Similar Items:
   - Given a target anime title, find users who rated it highly (≥ 8.0), collect their other items,
     infer meta predictions per user-item (using a chosen meta model), average per item, and return Top-N.
   - For display only, predicted ratings are clipped to [1, 10]; training/evaluation for RMSE stays raw.

Data Inputs
-----------
- meta_preprocessed.csv : item features (e.g., TF-IDF vectors) + a 'MAL_ID' column.
- rating_complete.csv   : training ratings dataframe with columns [user_id, anime_id, rating].
- rating_test.csv       : held-out test ratings (not used below but loaded for convenience).
- anime.csv             : mapping table with columns [MAL_ID, Name] to print human-readable titles.

Key Design Choices
------------------
- CB profile uses only Top-K neighbors per liked item before averaging.
- RMSE uses raw predictions; ranking and display use clipped predictions.

Hyperparameters (Meta-Learners)
-------------------------------
XGBoost:
- n_estimators=300, learning_rate=0.05, max_depth=6
- subsample=0.8, colsample_bytree=0.8
- objective="reg:squarederror", tree_method="hist"

LightGBM:
- n_estimators=400, learning_rate=0.03, max_depth=-1
- subsample=0.9, colsample_bytree=0.9
- objective="regression"

CatBoost:
- iterations=500, learning_rate=0.03, depth=6
- l2_leaf_reg=8.0, bagging_temperature=0.5, random_strength=2.0
- loss_function="RMSE", verbose=0
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# ------------------------------------------------------
# 1) Load Data
# ------------------------------------------------------
meta = pd.read_csv("meta_preprocessed.csv")
ratings_train = pd.read_csv("rating_complete.csv")
ratings_test = pd.read_csv("rating_test.csv")  # Not directly used below
anime_info = pd.read_csv("anime.csv")

meta = meta.fillna(0.0)
ratings_train = ratings_train[ratings_train['rating'] > 0]
ratings_test = ratings_test[ratings_test['rating'] > 0]

print(f"Train: {ratings_train.shape}, Test: {ratings_test.shape}\n")

# ------------------------------------------------------
# 2) CF (Surprise SVD)
# ------------------------------------------------------
reader = Reader(rating_scale=(ratings_train['rating'].min(), ratings_train['rating'].max()))
data = Dataset.load_from_df(ratings_train[['user_id', 'anime_id', 'rating']], reader)
trainset = data.build_full_trainset()

svd = SVD(
    n_factors=100,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42,
    verbose=True
)
svd.fit(trainset)
print("CF (SVD) training done!\n")

# ------------------------------------------------------
# 2-1) Additional Learning: NeuMF
# ------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks

train_df = pd.read_csv('rating_complete.csv')
test_df = pd.read_csv('rating.csv')
meta_train = pd.read_csv('meta_preprocessed.csv')
meta_test = pd.read_csv('meta_preprocessed_test.csv')

train_df = train_df[train_df['rating'] > 0]
test_df = test_df[test_df['rating'] > 0]

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

user_ids = train_df['user_id'].unique()
item_ids = train_df['anime_id'].unique()

user2idx = {u: i for i, u in enumerate(user_ids)}
item2idx = {a: i for i, a in enumerate(item_ids)}

train_df['user_idx'] = train_df['user_id'].map(user2idx)
train_df['item_idx'] = train_df['anime_id'].map(item2idx)

num_users = len(user2idx)
num_items = len(item2idx)

print(f"NeuMF User={num_users}, Item={num_items}")
embed_dim = 64
user_input = layers.Input(shape=(1,), name='user_input')
item_input = layers.Input(shape=(1,), name='item_input')

user_embed = layers.Embedding(num_users, embed_dim, name='user_embed')(user_input)
item_embed = layers.Embedding(num_items, embed_dim, name='item_embed')(item_input)
user_vec = layers.Flatten()(user_embed)
item_vec = layers.Flatten()(item_embed)

# GMF branch
gmf = layers.multiply([user_vec, item_vec])

# MLP branch
mlp = layers.concatenate([user_vec, item_vec])
mlp = layers.Dense(128, activation='relu')(mlp)
mlp = layers.BatchNormalization()(mlp)
mlp = layers.Dropout(0.3)(mlp)
mlp = layers.Dense(64, activation='relu')(mlp)
mlp = layers.BatchNormalization()(mlp)
mlp = layers.Dropout(0.2)(mlp)
mlp = layers.Dense(32, activation='relu')(mlp)
mlp = layers.Dropout(0.2)(mlp)

# Merge branches
concat = layers.concatenate([gmf, mlp])
output = layers.Dense(1, activation='linear')(concat)

neuMF = Model(inputs=[user_input, item_input], outputs=output)

neuMF.compile(
    optimizer=optimizers.Adam(learning_rate=0.0005),
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

train_users = train_df['user_idx'].values
train_items = train_df['item_idx'].values
train_ratings = train_df['rating'].values

train_u, val_u, train_i, val_i, train_r, val_r = train_test_split(
    train_users, train_items, train_ratings,
    test_size=0.2, random_state=42
)
print(f"\nTrain samples: {len(train_u)}, Validation samples: {len(val_u)}")
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,  # stop if no improvement for 3 epochs
    restore_best_weights=True,
    verbose=1
)
checkpoint = callbacks.ModelCheckpoint(
    'best_neumf.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=0
)
history = neuMF.fit(
    [train_u, train_i], train_r,
    validation_data=([val_u, val_i], val_r),
    epochs=20,
    batch_size=512,
    verbose=1,
    callbacks=[early_stop, checkpoint]
)

#validation_set RMSE
print(f"   Val Loss: {history.history['val_loss'][-1]:.4f}")
print(f"   Val RMSE: {history.history['val_root_mean_squared_error'][-1]:.4f}")

#test_set RMSE
def print_rmse(model, test_users, test_items, test_ratings, model_name="Model"):
    pred = model.predict([test_users, test_items], verbose=0).flatten()
    rmse = np.sqrt(np.mean((pred - test_ratings) ** 2))
    mae = np.mean(np.abs(pred - test_ratings))
    
    print(f"{model_name} test set \n | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    return rmse, mae

rmse, mae = print_rmse(neuMF, test_users, test_items, test_ratings, "NeuMF-Only")
# ------------------------------------------------------
# 3) Content Features & Item-Item Similarity
# ------------------------------------------------------
feature_cols = [c for c in meta.columns if c != 'MAL_ID']
scaler = StandardScaler()
X = scaler.fit_transform(meta[feature_cols].values)

malid_to_idx = {aid: i for i, aid in enumerate(meta['MAL_ID'].values)}
idx_to_malid = {v: k for k, v in malid_to_idx.items()}

print("Computing full item-item cosine similarity...")
item_sim_matrix = cosine_similarity(X)  # shape: (n_items, n_items)
print("Item similarity matrix ready!\n")

# ------------------------------------------------------
# 4) CF Score Helper
# ------------------------------------------------------
def cf_score(user_id: int, anime_id: int) -> float:
    """Predict CF score via SVD (est). Returns np.nan if missing."""
    try:
        return svd.predict(user_id, anime_id).est
    except Exception:
        return np.nan

# ------------------------------------------------------
# 5) CB Score with Top-K Neighbors per Liked Item
# ------------------------------------------------------
def get_cb_scores_for_user(user_id: int, like_th: float = 8.0, top_k: int = 30) -> np.ndarray | None:
    """
    Build a content-based profile vector for a user:
    1) Collect items the user liked (rating ≥ like_th).
    2) For each liked item, keep only Top-K most similar neighbors from the item-item matrix.
    3) Average those Top-K sims across liked items -> a "CB preference" score per item.
    Returns:
        np.ndarray of length n_items (scores aligned to meta order) or None if user has no liked items.
    """
    user_train = ratings_train[ratings_train['user_id'] == user_id]
    liked = user_train[user_train['rating'] >= like_th]['anime_id'].values
    liked_idx = [malid_to_idx[a] for a in liked if a in malid_to_idx]
    if not liked_idx:
        return None

    liked_sims = item_sim_matrix[liked_idx]  # (liked_count, n_items)

    # Mask to keep Top-K per liked item
    mask = np.zeros_like(liked_sims, dtype=bool)
    for i in range(len(liked_idx)):
        top_idx = np.argsort(-liked_sims[i])[:top_k]
        mask[i, top_idx] = True

    # Keep only Top-K entries, average across liked items.
    filtered = np.where(mask, liked_sims, 0.0)
    denom = np.maximum(mask.sum(axis=0), 1e-8)  # avoid div by zero
    cb_vector = filtered.sum(axis=0) / denom  # shape: (n_items,)

    return cb_vector

# ------------------------------------------------------
# 6) Build Meta-Train (sampled users for speed)
# ------------------------------------------------------
np.random.seed(42)
all_users = ratings_train['user_id'].unique()
sample_users = np.random.choice(all_users, size=min(3000, len(all_users)), replace=False)

meta_train_records: list[tuple[int, int, float, float, float]] = []
print(f"Sampling {len(sample_users)} / {len(all_users)} users for CB computation\n")

for uid in tqdm(sample_users, desc="Building Meta-Train"):
    user_data = ratings_train[ratings_train['user_id'] == uid]
    cb_vector = get_cb_scores_for_user(uid)
    if cb_vector is None:
        continue
    for row in user_data.itertuples(index=False):
        aid = row.anime_id
        if aid not in malid_to_idx:
            continue
        cb = cb_vector[malid_to_idx[aid]]   # CB score
        cf = cf_score(uid, aid)             # CF score
        meta_train_records.append((uid, aid, cf, cb, row.rating))

meta_train_df = pd.DataFrame(
    meta_train_records,
    columns=["user_id", "anime_id", "cf_score", "cb_score", "true_rating"]
)

# ------------------------------------------------------
# 7) Save Meta-Train
# ------------------------------------------------------
meta_train_df.to_csv("meta_train_ready.csv", index=False)
print("\nSaved meta_train_ready.csv")
print(meta_train_df.head())
print(f"\nMeta-Train built: {meta_train_df.shape[0]} samples")
print(meta_train_df.describe())

# ======================================================
# 8) Meta-Learners: XGB vs LGB vs Cat (raw predictions)
# ======================================================
meta_df = pd.read_csv("meta_train_ready.csv")  # reload to be explicit

X_meta = meta_df[["cf_score", "cb_score"]]
y_meta = meta_df["true_rating"]
X_train, X_val, y_train, y_val = train_test_split(
    X_meta, y_meta, test_size=0.2, random_state=42
)

# ---- XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist"
)

# ---- LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="regression",
    random_state=42
)

# ---- CatBoost
cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=8.0,
    bagging_temperature=0.5,
    random_strength=2.0,
    loss_function="RMSE",
    verbose=0,
    random_seed=42
)

print("\nTraining XGBoost...")
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("XGBoost done!")

print("Training LightGBM...")
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
print("LightGBM done!")

print("Training CatBoost...")
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
print("CatBoost done!\n")


# ------------------------------------------------------
# 9) RMSE (raw predictions) and P@10 / R@10 (CLIPPED for ranking)
# ------------------------------------------------------
def rmse(model, Xv, yv) -> float:
    pred = model.predict(Xv)                 # RAW for RMSE
    return float(np.sqrt(mean_squared_error(yv, pred)))

xgb_rmse = rmse(xgb_model, X_val, y_val)
lgb_rmse = rmse(lgb_model, X_val, y_val)
cat_rmse = rmse(cat_model, X_val, y_val)

LIKE_TH = 8.0
K = 10

# ======================================================
# 9-1) Select Best Model and Save Only 2 Files
# ======================================================
print("=" * 60)
print("MODEL SELECTION AND SAVE")
print("=" * 60)

# ------------------------------------------------------
# Compare models based on RMSE and select the best one
# ------------------------------------------------------
models_comparison = [
    ("XGBoost", xgb_model, xgb_rmse),
    ("LightGBM", lgb_model, lgb_rmse),
    ("CatBoost", cat_model, cat_rmse)
]

# Select model with the lowest RMSE value
best_name, best_model, best_rmse = min(models_comparison, key=lambda x: x[2])

print(f"\nModel Performance:")
print(f"  XGBoost:  RMSE = {xgb_rmse:.4f}")
print(f"  LightGBM: RMSE = {lgb_rmse:.4f}")
print(f"  CatBoost: RMSE = {cat_rmse:.4f}")
print(f"\nBest Model: {best_name} (RMSE = {best_rmse:.4f})")

# ------------------------------------------------------
# Create models directory if it does not exist
# ------------------------------------------------------
os.makedirs("models", exist_ok=True)

# ======================================================
# 9-2) Save File 1: svd_model.pkl
# ======================================================
print("\nSaving svd_model.pkl...")

# Package SVD model and parameters
svd_package = {
    'model': svd,  # Trained SVD model object
    'params': {
        'n_factors': 100,   # Number of latent factors
        'n_epochs': 20,     # Number of training epochs
        'lr_all': 0.005,    # Learning rate for all parameters
        'reg_all': 0.02     # Regularization strength
    }
}

# Save as pickle file
with open("models/svd_model.pkl", "wb") as f:
    pickle.dump(svd_package, f)
print("   Saved: models/svd_model.pkl")

# ======================================================
# 9-3) Save File 2: meta_model.pkl
# ======================================================
print("\nSaving meta_model.pkl...")

# Package meta-model and relevant metadata
meta_package = {
    'model': best_model,                   # Trained meta-learner (XGBoost, LGBM, or CatBoost)
    'model_type': best_name,               # Model name
    'rmse': best_rmse,                     # Validation RMSE
    'scaler': scaler,                      # Feature scaler used during training
    'malid_to_idx': malid_to_idx,          # Mapping: MAL anime ID -> matrix index
    'idx_to_malid': idx_to_malid,          # Mapping: matrix index -> MAL anime ID
    'item_sim_matrix': item_sim_matrix,    # Item-item similarity matrix for CB
    'feature_cols': feature_cols,          # Feature columns used in meta-training
    'training_params': {                   # Metadata about training configuration
        'like_threshold': 8.0,
        'top_k_neighbors': 30,
        'sample_users': len(sample_users),
        'total_users': len(all_users)
    }
}

# Save as pickle file
with open("models/meta_model.pkl", "wb") as f:
    pickle.dump(meta_package, f)
print("   Saved: models/meta_model.pkl")

# ======================================================
# 9-4) Summary of Saved Models
# ======================================================
print("\n" + "=" * 60)
print("MODEL SAVE COMPLETE!")
print("=" * 60)
print("\nSaved files:")
print("  1. models/svd_model.pkl    - SVD collaborative filtering model")
print("  2. models/meta_model.pkl   - Best meta-learner + preprocessing package")
print()

# ------------------------------------------------------
# Display file sizes for confirmation
# ------------------------------------------------------
svd_size = os.path.getsize("models/svd_model.pkl") / (1024 * 1024)
meta_size = os.path.getsize("models/meta_model.pkl") / (1024 * 1024)

print("File sizes:")
print(f"  svd_model.pkl:  {svd_size:.2f} MB")
print(f"  meta_model.pkl: {meta_size:.2f} MB")


def precision_recall_at_k(model, df: pd.DataFrame, k: int = 10, like_th: float = 8.0):
    """
    Compute mean Precision@K and Recall@K across users with at least one relevant item.
    Uses predictions CLIPPED to [1, 10] for ranking to avoid overshoot-driven distortions.
    """
    precisions, recalls = [], []
    for uid, group in df.groupby("user_id"):
        g = group.copy()
        raw_pred = model.predict(g[["cf_score", "cb_score"]])
        g["meta_pred"] = np.clip(raw_pred, 1.0, 10.0)   # CLIPPED for ranking
        topk = g.sort_values("meta_pred", ascending=False).head(k)
        relevant = g[g["true_rating"] >= like_th]
        if len(relevant) == 0:
            continue
        hits = len(set(topk["anime_id"]) & set(relevant["anime_id"]))
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))
    return float(np.mean(precisions)), float(np.mean(recalls)), len(precisions)

xgb_prec, xgb_rec, n_users = precision_recall_at_k(xgb_model, meta_df, K, LIKE_TH)
lgb_prec, lgb_rec, _ = precision_recall_at_k(lgb_model, meta_df, K, LIKE_TH)
cat_prec, cat_rec, _ = precision_recall_at_k(cat_model, meta_df, K, LIKE_TH)

print("\nModel Performance Comparison\n")
print(f"{'Model':<12} | {'RMSE(raw)':<10} | {'Precision@10':<13} | {'Recall@10':<10}")
print("-" * 65)
print(f"{'XGBoost':<12} | {xgb_rmse:<10.4f} | {xgb_prec:<13.4f} | {xgb_rec:<10.4f}")
print(f"{'LightGBM':<12} | {lgb_rmse:<10.4f} | {lgb_prec:<13.4f} | {lgb_rec:<10.4f}")
print(f"{'CatBoost':<12} | {cat_rmse:<10.4f} | {cat_prec:<13.4f} | {cat_rec:<10.4f}")
print("-" * 65)
print(f"Users evaluated: {n_users}")

winner = min(
    [("XGBoost", xgb_rmse), ("LightGBM", lgb_rmse), ("CatBoost", cat_rmse)],
    key=lambda x: x[1]
)[0]
print(f"\nLowest RMSE model: {winner}\n")

# ------------------------------------------------------
# 10) Side-by-side Top-10 for a Sample User
#       - Sort by RAW predictions (unchanged)
#       - Display CLIPPED values in print only
# ------------------------------------------------------
sample_user = meta_df["user_id"].sample(1, random_state=42).iloc[0]
user_data = meta_df[meta_df["user_id"] == sample_user].copy()

# RAW predictions for sorting
user_data["xgb_pred_raw"] = xgb_model.predict(user_data[["cf_score", "cb_score"]])
user_data["lgb_pred_raw"] = lgb_model.predict(user_data[["cf_score", "cb_score"]])
user_data["cat_pred_raw"] = cat_model.predict(user_data[["cf_score", "cb_score"]])

# CLIPPED for display
user_data["xgb_pred_disp"] = np.clip(user_data["xgb_pred_raw"], 1.0, 10.0)
user_data["lgb_pred_disp"] = np.clip(user_data["lgb_pred_raw"], 1.0, 10.0)
user_data["cat_pred_disp"] = np.clip(user_data["cat_pred_raw"], 1.0, 10.0)

user_data = user_data.merge(
    anime_info[["MAL_ID", "Name"]],
    left_on="anime_id",
    right_on="MAL_ID",
    how="left"
)

top10_xgb = user_data.sort_values("xgb_pred_raw", ascending=False).head(10)
top10_lgb = user_data.sort_values("lgb_pred_raw", ascending=False).head(10)
top10_cat = user_data.sort_values("cat_pred_raw", ascending=False).head(10)

print(f"User {sample_user} — Top-10 by three models")
print("(Sorted by RAW predictions; values displayed are CLIPPED to [1,10])\n" + "-"*120)
print(f"{'XGBoost Top':<38} | {'pred/true':<10} || {'LightGBM Top':<38} | {'pred/true':<10} || {'CatBoost Top':<38} | {'pred/true'}")
print("-"*120)
for i in range(10):
    nx, px, tx = top10_xgb.iloc[i]["Name"], top10_xgb.iloc[i]["xgb_pred_disp"], top10_xgb.iloc[i]["true_rating"]
    nl, pl, tl = top10_lgb.iloc[i]["Name"], top10_lgb.iloc[i]["lgb_pred_disp"], top10_lgb.iloc[i]["true_rating"]
    nc, pc, tc = top10_cat.iloc[i]["Name"], top10_cat.iloc[i]["cat_pred_disp"], top10_cat.iloc[i]["true_rating"]
    print(f"{nx[:36]:<38} | {px:>4.2f}/{tx:<4.1f} || {nl[:36]:<38} | {pl:>4.2f}/{tl:<4.1f} || {nc[:36]:<38} | {pc:>4.2f}/{tc:<4.1f}")

# ======================================================
# 11) Model-based "Similar Items" Recommender (by Meta model)
#       - Display uses CLIPPED predictions
# ======================================================
def recommend_similar_by_model(
    target_title: str,
    top_n: int = 10,
    meta_model=None,
    model_name: str = "LGBM",
    like_th: float = 8.0
):
    """
    Recommend items that users who liked `target_title` are also likely to like.

    Steps:
    1) Find target item by title (partial match).
    2) Collect users who rated the target item ≥ like_th.
    3) Gather all (user, other_item) pairs those users rated.
    4) Merge with meta features (cf_score, cb_score); predict meta rating with `meta_model`.
    5) Average predictions per item and return top-N.
    Notes:
    - Predictions are CLIPPED to [1, 10] for display only.
    """
    anime_df = pd.read_csv("anime.csv")
    ratings_df = pd.read_csv("rating_complete.csv")
    meta_ready = pd.read_csv("meta_train_ready.csv")

    # 1) Target item lookup
    match = anime_df[anime_df["Name"].str.contains(target_title, case=False, na=False)]
    if match.empty:
        print(f"Title '{target_title}' not found.")
        return
    target_id = match.iloc[0]["MAL_ID"]
    target_name = match.iloc[0]["Name"]
    print(f"\nRecommendations conditioned on '{target_name}' (MAL_ID={target_id})\n")

    # 2) Users who liked target
    liked_users = ratings_df[(ratings_df["anime_id"] == target_id) & (ratings_df["rating"] >= like_th)]["user_id"].unique()
    if len(liked_users) == 0:
        print("No users liked the target item.")
        return

    # 3) Candidate (user, item) among those users excluding target
    candidate = ratings_df[ratings_df["user_id"].isin(liked_users)]
    candidate = candidate[candidate["anime_id"] != target_id]
    candidate = candidate.merge(anime_df[["MAL_ID", "Name"]], left_on="anime_id", right_on="MAL_ID", how="left")

    # 4) Merge CF/CB scores; predict meta rating
    merged = candidate.merge(
        meta_ready[["user_id", "anime_id", "cf_score", "cb_score"]],
        on=["user_id", "anime_id"],
        how="left"
    ).dropna(subset=["cf_score", "cb_score"])
    if merged.empty:
        print("No candidates with available CF/CB scores.")
        return

    preds = meta_model.predict(merged[["cf_score", "cb_score"]])
    merged["meta_pred"] = np.clip(preds, 1.0, 10.0)  # display-only clipping

    # 5) Average per item and rank
    item_scores = (
        merged.groupby(["anime_id", "Name"])["meta_pred"]
        .mean()
        .reset_index()
        .sort_values("meta_pred", ascending=False)
    )
    item_scores = item_scores[item_scores["anime_id"] != target_id].head(top_n)

    print(f"Items users who liked '{target_name}' are also likely to like — Top-{top_n} ({model_name})")
    print("-" * 100)
    for _, row in item_scores.iterrows():
        print(f"{row['Name']:<70} | Predicted: {row['meta_pred']:.2f}")

# Example calls (choose the meta model you prefer):
recommend_similar_by_model("Koe no Katachi", top_n=10, meta_model=lgb_model, model_name="LightGBM")
recommend_similar_by_model("Koe no Katachi", top_n=10, meta_model=xgb_model, model_name="XGBoost")
recommend_similar_by_model("Koe no Katachi", top_n=10, meta_model=cat_model, model_name="CatBoost")


#=========================================================
# Interactive 3D t-SNE Visualization of Anime Embeddings
# ==========================================================
# This script visualizes anime metadata using 3D t-SNE based on TF-IDF features.
# Each point represents an anime, colored by its top genre.
#
# Libraries:
# - plotly.express: for interactive 3D visualization
# - sklearn.manifold.TSNE: for nonlinear dimensionality reduction
# - sklearn.decomposition.PCA: for initial dimensionality reduction before t-SNE
# - sklearn.preprocessing.StandardScaler: for feature scaling
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ----------------------------------------------------------
# 1) Load and Merge Data
# ----------------------------------------------------------
# Load preprocessed metadata and original anime information.
# Merge to include textual genre information for labeling clusters.
meta = pd.read_csv("meta_preprocessed.csv")
anime_info = pd.read_csv("anime.csv")

# Merge based on MAL_ID to recover genre information
merged = meta.merge(anime_info[["MAL_ID", "Genres"]], on="MAL_ID", how="left")

# ----------------------------------------------------------
# 2) Extract TF-IDF Feature Subset
# ----------------------------------------------------------
# Select TF-IDF columns for Genres, Studios, and Producers.
# These represent semantic text-based content features.
genre_cols = [c for c in merged.columns if c.startswith("Genre_")]
studio_cols = [c for c in merged.columns if c.startswith("Studio_")]
prod_cols   = [c for c in merged.columns if c.startswith("Prod_")]

# Stack all TF-IDF features together
tfidf_features = merged[genre_cols + studio_cols + prod_cols].values

# Scale features for stability before PCA and t-SNE
scaler = StandardScaler(with_mean=False)
tfidf_scaled = scaler.fit_transform(tfidf_features)

# ----------------------------------------------------------
# 3) PCA (50D) + t-SNE (3D)
# ----------------------------------------------------------
# PCA reduces high-dimensional TF-IDF vectors to 50D to remove noise
# t-SNE then projects these into a 3D embedding for visualization.
pca_50 = PCA(n_components=50, random_state=42).fit_transform(tfidf_scaled)

tsne_3d = TSNE(
    n_components=3,       # output 3D coordinates
    perplexity=30,        # balance between local/global structure
    max_iter=1500,        # number of optimization iterations
    learning_rate=200,    # gradient step size
    random_state=42,      # reproducibility
    verbose=True
)
emb_3d = tsne_3d.fit_transform(pca_50)

# Add embedding coordinates and main genre to dataframe
merged["x"] = emb_3d[:, 0]
merged["y"] = emb_3d[:, 1]
merged["z"] = emb_3d[:, 2]
merged["TopGenre"] = merged["Genres"].apply(
    lambda g: g.split(",")[0].strip() if isinstance(g, str) and g else "Unknown"
)

# ----------------------------------------------------------
# 4) Sampling (optional for performance)
# ----------------------------------------------------------
# Randomly sample up to 3000 anime for faster rendering.
sample_df = merged.sample(min(3000, len(merged)), random_state=42)

# ----------------------------------------------------------
# 5) Interactive 3D Plot with Plotly
# ----------------------------------------------------------
# Each point corresponds to an anime embedding colored by its top genre.
fig = px.scatter_3d(
    sample_df,
    x="x", y="y", z="z",
    color="TopGenre",
    hover_data=["MAL_ID"],
    title="Interactive 3D t-SNE of Anime TF-IDF Embeddings",
    color_discrete_sequence=px.colors.qualitative.Set3
)

# Customize marker appearance and plot aesthetics
fig.update_traces(marker=dict(size=4, opacity=0.8))
fig.update_layout(
    legend=dict(title="Top Genre", itemsizing="trace"),
    margin=dict(l=0, r=0, b=0, t=40),
    scene=dict(
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        zaxis_title="t-SNE Dimension 3",
        bgcolor="white"
    )
)

# Display interactive 3D visualization
fig.show()


#=========================================================
# Visualizing Hybrid Meta-Learner Recommendations in 3D t-SNE
# ==========================================================
# This script visualizes hybrid meta-learner recommendation results
# in a 3D t-SNE embedding space. It highlights:
#   - All anime embeddings (gray background)
#   - Watched anime by a sampled user (blue dots)
#   - Recommended anime by the meta-learner (gold stars)
#
# Libraries:
# - pandas, numpy: data manipulation and numerical computation
# - sklearn.manifold.TSNE: nonlinear dimensionality reduction (visual embedding)
# - sklearn.decomposition.PCA: dimensionality reduction before t-SNE
# - sklearn.preprocessing.StandardScaler: feature scaling
# - plotly.graph_objects: 3D interactive visualization
# ==========================================================

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# ----------------------------------------------------------
# 1) Load all relevant data
# ----------------------------------------------------------
# Load anime metadata, rating history, and precomputed meta-features.
meta = pd.read_csv("meta_preprocessed.csv")
anime_info = pd.read_csv("anime.csv")
ratings = pd.read_csv("rating_complete.csv")
meta_ready = pd.read_csv("meta_train_ready.csv")

# Remove adult content for presentation clarity
anime_info = anime_info[~anime_info["Genres"].fillna("").str.contains("Hentai", case=False)]

# Merge TF-IDF metadata with anime genre and name information
merged = meta.merge(anime_info[["MAL_ID", "Name", "Genres"]], on="MAL_ID", how="left")

# ----------------------------------------------------------
# 2) Select TF-IDF features
# ----------------------------------------------------------
# Extract all TF-IDF feature columns related to genres, studios, and producers.
genre_cols = [c for c in merged.columns if c.startswith("Genre_")]
studio_cols = [c for c in merged.columns if c.startswith("Studio_")]
prod_cols   = [c for c in merged.columns if c.startswith("Prod_")]

# Combine all selected feature vectors
tfidf_features = merged[genre_cols + studio_cols + prod_cols].values

# Normalize feature magnitudes to improve PCA and t-SNE stability
scaler = StandardScaler(with_mean=False)
tfidf_scaled = scaler.fit_transform(tfidf_features)

# Optionally sample for computational efficiency
if len(tfidf_scaled) > 6000:
    merged = merged.sample(6000, random_state=42)
    tfidf_scaled = tfidf_scaled[merged.index]

# ----------------------------------------------------------
# 3) 3D t-SNE Embedding
# ----------------------------------------------------------
# First, reduce TF-IDF features from high dimensions to 50 using PCA.
# Then, apply t-SNE to obtain a 3D representation for visualization.
pca_50 = PCA(n_components=50, random_state=42).fit_transform(tfidf_scaled)
tsne_3d = TSNE(
    n_components=3,       # Project data into 3D space
    perplexity=30,        # Controls the balance between local/global structure
    max_iter=1200,        # Number of optimization iterations
    learning_rate=200,    # Step size for gradient descent
    random_state=42,      # Reproducibility
    verbose=True
)
emb_3d = tsne_3d.fit_transform(pca_50)

# Add the t-SNE coordinates and top genre labels to the dataframe
merged["x"], merged["y"], merged["z"] = emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2]
merged["TopGenre"] = merged["Genres"].apply(
    lambda g: g.split(",")[0].strip() if isinstance(g, str) and g else "Unknown"
)

# ----------------------------------------------------------
# 4) Pick sample user and generate hybrid recommendations
# ----------------------------------------------------------
# Select one random user for demonstration
sample_user = meta_ready["user_id"].sample(1, random_state=42).iloc[0]
user_data = meta_ready[meta_ready["user_id"] == sample_user].copy()

# Generate meta-learner predictions using the trained LightGBM model
user_data["meta_pred"] = lgb_model.predict(user_data[["cf_score", "cb_score"]])

# Retrieve top 10 recommended anime based on predicted ratings
recommended_ids = user_data.sort_values("meta_pred", ascending=False).head(10)["anime_id"].values

# Retrieve anime the user has already watched (rating ≥ 7)
watched_df = ratings[(ratings["user_id"] == sample_user) & (ratings["rating"] >= 7)]
watched_ids = watched_df["anime_id"].unique()

# ----------------------------------------------------------
# 5) User Preference Summary (Top 10 liked anime)
# ----------------------------------------------------------
# Print the user's top-rated anime and their genres
user_liked = (
    watched_df.merge(anime_info, left_on="anime_id", right_on="MAL_ID", how="left")
    .sort_values("rating", ascending=False)
    .head(10)[["Name", "Genres", "rating"]]
)

print("=" * 70)
print(f"User {sample_user}'s Favorite Anime (Top 10 by Rating ≥ 7)")
print("=" * 70)
for idx, row in user_liked.iterrows():
    print(f"{idx+1:2d}. {row['Name']} — {row['Genres']} (Rating: {row['rating']})")

# Identify top genres preferred by the user
print("\nTop Genres Preferred:")
genre_series = (
    user_liked["Genres"].dropna().str.split(",").explode().str.strip().value_counts().head(5)
)
print(genre_series)

# ----------------------------------------------------------
# 6) Prepare Data for Visualization
# ----------------------------------------------------------
# Separate embeddings into background, watched, and recommended subsets
background = merged[
    ~merged["MAL_ID"].isin(set(watched_ids) | set(recommended_ids))
]
watched = merged[merged["MAL_ID"].isin(watched_ids)]
recommended = merged[merged["MAL_ID"].isin(recommended_ids)]

# ----------------------------------------------------------
# 7) Plot Interactive 3D Visualization
# ----------------------------------------------------------
fig = go.Figure()

# Plot all anime as gray background points
fig.add_trace(go.Scatter3d(
    x=background["x"], y=background["y"], z=background["z"],
    mode="markers",
    marker=dict(size=3, color="lightgray", opacity=0.3),
    name="All Anime"
))

# Plot watched anime as blue circles
fig.add_trace(go.Scatter3d(
    x=watched["x"], y=watched["y"], z=watched["z"],
    mode="markers",
    marker=dict(size=6, color="dodgerblue", opacity=0.8, symbol="circle"),
    text=[f"{n}<br>Genre: {g}" for n, g in zip(watched["Name"], watched["TopGenre"])],
    hoverinfo="text",
    name="Watched by User"
))

# Plot recommended anime as gold stars
fig.add_trace(go.Scatter3d(
    x=recommended["x"], y=recommended["y"], z=recommended["z"],
    mode="text+markers",
    text=["★"] * len(recommended),
    textfont=dict(size=20, color="gold"),
    marker=dict(size=10, color="gold", opacity=0.9, symbol="diamond"),
    hovertext=[
        f"Recommended: {n}<br>Genre: {g}<br>Predicted Rating: {p:.2f}"
        for n, g, p in zip(
            recommended["Name"],
            recommended["TopGenre"],
            user_data.sort_values("meta_pred", ascending=False).head(10)["meta_pred"]
        )
    ],
    hoverinfo="text",
    name="Recommended"
))

# Configure layout and aesthetics
fig.update_layout(
    title=f"3D t-SNE Embedding — User {sample_user}'s Hybrid Recommendations",
    scene=dict(
        xaxis_title="t-SNE Dim 1",
        yaxis_title="t-SNE Dim 2",
        zaxis_title="t-SNE Dim 3",
        bgcolor="white"
    ),
    legend=dict(
        title="Legend",
        itemsizing="trace",
        yanchor="top", y=0.98, xanchor="left", x=0.02
    ),
    margin=dict(l=0, r=0, b=0, t=60)
)

# Render the 3D visualization
fig.show()


