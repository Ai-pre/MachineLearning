# ======================================================
# ⚡ CF(SVD) + CB(Content-Based) Hybrid Meta-Train Builder
#     + XGBoost vs LightGBM vs CatBoost (raw predictions)
#     + Model-based "similar items" recommender
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
   - Evaluation uses raw predictions (no scaling) for RMSE and ranking.
     * Rationale: the models already learn the 1–10 target distribution; extra scaling can distort rank order.

5) Ranking Metrics:
   - Precision@K, Recall@K using raw predictions. A rating ≥ 8.0 is considered "relevant".

6) Model-based Similar Items:
   - Given a target anime title, find users who rated it highly (≥ 8.0), collect their other items,
     infer meta predictions per user-item (using a chosen meta model), average per item, and return Top-N.
   - For display only, we clip predicted ratings to [1, 10] to avoid overshoot in printed values,
     but raw predictions are used everywhere else in training/evaluation.

Data Inputs
-----------
- meta_preprocessed.csv : item features (e.g., TF-IDF vectors) + a 'MAL_ID' column.
- rating_complete.csv   : training ratings dataframe with columns [user_id, anime_id, rating].
- rating_test.csv       : held-out test ratings (not used below but loaded for convenience).
- anime.csv             : mapping table with columns [MAL_ID, Name] to print human-readable titles.

Key Design Choices
------------------
- CB profile uses only Top-K neighbors per liked item before averaging:
  This reduces popularity bleed and focuses the profile on sharp, strongest signals.

- Raw predictions for RMSE/Ranking:
  Tree ensembles (especially LGBM/CatBoost) naturally calibrate to the target range;
  extra min-max scaling can worsen RMSE and distort rank order.

Hyperparameters (Meta-Learners)
-------------------------------
XGBoost:
- n_estimators=300: moderate number of trees (increase for potentially lower bias).
- learning_rate=0.05: smaller step for smoother function; balances bias/variance.
- max_depth=6: controls leaf complexity; deeper trees risk overfitting and overshoot.
- subsample=0.8, colsample_bytree=0.8: stochasticity for regularization.
- objective="reg:squarederror": standard L2 regression for RMSE optimization.
- tree_method="hist": faster histogram-based construction.

LightGBM:
- n_estimators=400, learning_rate=0.03: slightly more trees with smaller steps.
- max_depth=-1: let LightGBM decide best depth (use with care; regularized by subsampling).
- subsample=0.9, colsample_bytree=0.9: robust generalization via sampling.
- objective="regression": L2 loss (RMSE).

CatBoost:
- iterations=500, learning_rate=0.03, depth=6: balanced capacity and step size.
- l2_leaf_reg=8.0: stronger L2 regularization to reduce overshoot.
- bagging_temperature=0.5: moderate Bayesian bagging strength.
- random_strength=2.0: random noise in splits to improve generalization.

Metrics
-------
- RMSE on validation split (raw predictions).
- Precision@10 / Recall@10 averaged across users with ≥1 relevant item (rating ≥ 8.0).

Notes
-----
- If you observe consistent overshoot (>10) in raw predictions, tighten regularization
  (e.g., increase L2, reduce max_depth or learning_rate) rather than post-scaling,
  to preserve rank ordering quality.
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

print(f"✅ Train: {ratings_train.shape}, Test: {ratings_test.shape}\n")

# ------------------------------------------------------
# 2) CF (Surprise SVD)
# ------------------------------------------------------
reader = Reader(rating_scale=(ratings_train['rating'].min(), ratings_train['rating'].max()))
data = Dataset.load_from_df(ratings_train[['user_id', 'anime_id', 'rating']], reader)
trainset = data.build_full_trainset()

svd = SVD(
    n_factors=100,    # latent dimension
    n_epochs=20,      # training epochs
    lr_all=0.005,     # learning rate for biases and factors
    reg_all=0.02,     # L2 regularization for biases and factors
    random_state=42,
    verbose=True
)
svd.fit(trainset)
print("🎯 CF (SVD) training done!\n")

# ------------------------------------------------------
# 3) Content Features & Item-Item Similarity
# ------------------------------------------------------
feature_cols = [c for c in meta.columns if c != 'MAL_ID']
scaler = StandardScaler()
X = scaler.fit_transform(meta[feature_cols].values)

malid_to_idx = {aid: i for i, aid in enumerate(meta['MAL_ID'].values)}
idx_to_malid = {v: k for k, v in malid_to_idx.items()}

print("🧮 Computing full item-item cosine similarity...")
item_sim_matrix = cosine_similarity(X)  # shape: (n_items, n_items)
print("✅ Item similarity matrix ready!\n")

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
print(f"👥 Sampling {len(sample_users)} / {len(all_users)} users for CB computation\n")

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
print("\n💾 Saved meta_train_ready.csv")
print(meta_train_df.head())
print(f"\n📊 Meta-Train built: {meta_train_df.shape[0]} samples")
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

print("\n🎯 Training XGBoost...")
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("✅ XGBoost done!")

print("🎯 Training LightGBM...")
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
print("✅ LightGBM done!")

print("🎯 Training CatBoost...")
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
print("✅ CatBoost done!\n")

# ------------------------------------------------------
# 9) RMSE (raw predictions) and P@10 / R@10 (raw)
# ------------------------------------------------------
def rmse(model, Xv, yv) -> float:
    pred = model.predict(Xv)
    return float(np.sqrt(mean_squared_error(yv, pred)))

xgb_rmse = rmse(xgb_model, X_val, y_val)
lgb_rmse = rmse(lgb_model, X_val, y_val)
cat_rmse = rmse(cat_model, X_val, y_val)

LIKE_TH = 8.0
K = 10

def precision_recall_at_k(model, df: pd.DataFrame, k: int = 10, like_th: float = 8.0):
    """
    Compute mean Precision@K and Recall@K across users with at least one relevant item.
    Uses raw predictions for ranking (no scaling).
    """
    precisions, recalls = [], []
    for uid, group in df.groupby("user_id"):
        g = group.copy()
        g["meta_pred"] = model.predict(g[["cf_score", "cb_score"]])
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

print("\n📊 Model Performance Comparison (Raw Predictions)\n")
print(f"{'Model':<12} | {'RMSE(raw)':<10} | {'Precision@10':<13} | {'Recall@10':<10}")
print("-" * 65)
print(f"{'XGBoost':<12} | {xgb_rmse:<10.4f} | {xgb_prec:<13.4f} | {xgb_rec:<10.4f}")
print(f"{'LightGBM':<12} | {lgb_rmse:<10.4f} | {lgb_prec:<13.4f} | {lgb_rec:<10.4f}")
print(f"{'CatBoost':<12} | {cat_rmse:<10.4f} | {cat_prec:<13.4f} | {cat_rec:<10.4f}")
print("-" * 65)
print(f"👥 Users evaluated: {n_users}")

winner = min(
    [("XGBoost", xgb_rmse), ("LightGBM", lgb_rmse), ("CatBoost", cat_rmse)],
    key=lambda x: x[1]
)[0]
print(f"\n🏆 Lowest RMSE model: {winner}\n")

# ------------------------------------------------------
# 10) Side-by-side Top-10 for a Sample User (raw)
# ------------------------------------------------------
sample_user = meta_df["user_id"].sample(1, random_state=42).iloc[0]
user_data = meta_df[meta_df["user_id"] == sample_user].copy()

user_data["xgb_pred"] = xgb_model.predict(user_data[["cf_score", "cb_score"]])
user_data["lgb_pred"] = lgb_model.predict(user_data[["cf_score", "cb_score"]])
user_data["cat_pred"] = cat_model.predict(user_data[["cf_score", "cb_score"]])

user_data = user_data.merge(
    anime_info[["MAL_ID", "Name"]],
    left_on="anime_id",
    right_on="MAL_ID",
    how="left"
)

top10_xgb = user_data.sort_values("xgb_pred", ascending=False).head(10)
top10_lgb = user_data.sort_values("lgb_pred", ascending=False).head(10)
top10_cat = user_data.sort_values("cat_pred", ascending=False).head(10)

print(f"🎯 User {sample_user} — Top-10 by three models (No Scaling)\n{'-'*120}")
print(f"{'XGBoost Top':<38} | {'pred/true':<10} || {'LightGBM Top':<38} | {'pred/true':<10} || {'CatBoost Top':<38} | {'pred/true'}")
print("-"*120)
for i in range(10):
    nx, px, tx = top10_xgb.iloc[i]["Name"], top10_xgb.iloc[i]["xgb_pred"], top10_xgb.iloc[i]["true_rating"]
    nl, pl, tl = top10_lgb.iloc[i]["Name"], top10_lgb.iloc[i]["lgb_pred"], top10_lgb.iloc[i]["true_rating"]
    nc, pc, tc = top10_cat.iloc[i]["Name"], top10_cat.iloc[i]["cat_pred"], top10_cat.iloc[i]["true_rating"]
    print(f"{nx[:36]:<38} | {px:>4.2f}/{tx:<4.1f} || {nl[:36]:<38} | {pl:>4.2f}/{tl:<4.1f} || {nc[:36]:<38} | {pc:>4.2f}/{tc:<4.1f}")

# ======================================================
# 11) Model-based "Similar Items" Recommender (by Meta model)
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
    - We clip predictions only for display (1..10), not for training/evaluation elsewhere.
    """
    anime_df = pd.read_csv("anime.csv")
    ratings_df = pd.read_csv("rating_complete.csv")
    meta_ready = pd.read_csv("meta_train_ready.csv")

    # 1) Target item lookup
    match = anime_df[anime_df["Name"].str.contains(target_title, case=False, na=False)]
    if match.empty:
        print(f"❌ Title '{target_title}' not found.")
        return
    target_id = match.iloc[0]["MAL_ID"]
    target_name = match.iloc[0]["Name"]
    print(f"\n🎯 Recommendations conditioned on '{target_name}' (MAL_ID={target_id})\n")

    # 2) Users who liked target
    liked_users = ratings_df[(ratings_df["anime_id"] == target_id) & (ratings_df["rating"] >= like_th)]["user_id"].unique()
    if len(liked_users) == 0:
        print("❌ No users liked the target item.")
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
        print("❌ No candidates with available CF/CB scores.")
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

    print(f"📺 Items users who liked '{target_name}' are also likely to like — Top-{top_n} ({model_name})")
    print("-" * 100)
    for _, row in item_scores.iterrows():
        print(f"{row['Name']:<70} | Predicted: {row['meta_pred']:.2f}")

# Example calls (choose the meta model you prefer):
recommend_similar_by_model("Koe no Katachi", top_n=10, meta_model=lgb_model, model_name="LightGBM")
recommend_similar_by_model("Koe no Katachi", top_n=10, meta_model=xgb_model, model_name="XGBoost")
recommend_similar_by_model("Koe no Katachi", top_n=10, meta_model=cat_model, model_name="CatBoost")
