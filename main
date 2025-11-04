# ============================================
# meta_preprocessed.csv 생성 (CB용 Feature 강화 & Score 포함 스케일링)
# + TF-IDF / Encoder / Scaler 저장
# ============================================

import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# 1️⃣ 데이터 로드 및 결측 처리
# -----------------------------
meta_df = pd.read_csv("anime.csv")
print(f"✅ Loaded anime.csv, shape: {meta_df.shape}")

# 주요 결측값 채우기
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

# MAL_ID 정규화
meta_df['MAL_ID'] = meta_df['MAL_ID'].astype(int)

# -----------------------------
# 2️⃣ 범주형 인코딩 (저장 가능하도록 수정)
# -----------------------------
label_cols = ['Type', 'Source', 'Rating', 'Premiered', 'Duration']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    meta_df[f'{col}_encoded'] = le.fit_transform(meta_df[col].astype(str))
    label_encoders[col] = le

# -----------------------------
# 3️⃣ TF-IDF 피처 생성 (fit 객체도 함께 저장)
# -----------------------------
def make_tfidf_features(df, column, prefix):
    tfidf = TfidfVectorizer(
        token_pattern=r'[^, ]+',
        stop_words='english',
        max_features=100
    )
    mat = tfidf.fit_transform(df[column])
    tfidf_df = pd.DataFrame(
        mat.toarray(),
        columns=[f"{prefix}_{t}" for t in tfidf.get_feature_names_out()]
    )
    return tfidf_df, tfidf

tfidf_genres, vec_genres = make_tfidf_features(meta_df, 'Genres', 'Genre')
tfidf_producers, vec_producers = make_tfidf_features(meta_df, 'Producers', 'Prod')
tfidf_studios, vec_studios = make_tfidf_features(meta_df, 'Studios', 'Studio')

# -----------------------------
# 4️⃣ 수치형 컬럼 처리 (Score 포함)
# -----------------------------
numeric_cols = ['Score', 'Episodes', 'Ranked', 'Popularity', 'Members', 'Favorites']
for col in numeric_cols:
    meta_df[col] = pd.to_numeric(meta_df[col], errors='coerce')
meta_df[numeric_cols] = meta_df[numeric_cols].fillna(meta_df[numeric_cols].mean())

scaler = MinMaxScaler()
meta_df[numeric_cols] = scaler.fit_transform(meta_df[numeric_cols])

# -----------------------------
# 5️⃣ CB용 Feature 결합
# -----------------------------
meta_processed = pd.concat(
    [
        meta_df[['MAL_ID'] + numeric_cols + [f'{col}_encoded' for col in label_cols]],
        tfidf_genres,
        tfidf_producers,
        tfidf_studios
    ],
    axis=1
)

# -----------------------------
# 6️⃣ rating_complete.csv와 매칭되는 ID만 필터링
# -----------------------------
rating_df = pd.read_csv("rating_complete.csv")
rating_df['anime_id'] = rating_df['anime_id'].astype(int)

valid_ids = set(rating_df['anime_id']).intersection(set(meta_processed['MAL_ID']))
meta_processed = meta_processed[meta_processed['MAL_ID'].isin(valid_ids)].reset_index(drop=True)

print(f"✅ Matched items with rating data: {len(valid_ids)} / {len(meta_df)}")
print(f"✅ Final processed shape: {meta_processed.shape}")

# -----------------------------
# 7️⃣ 저장 (데이터 + 객체)
# -----------------------------
meta_processed.to_csv("meta_preprocessed.csv", index=False)
print("💾 Saved: meta_preprocessed.csv")

artifacts = {
    "label_encoders": label_encoders,
    "tfidf_genre": vec_genres,
    "tfidf_prod": vec_producers,
    "tfidf_studio": vec_studios,
    "scaler": scaler,
    "numeric_cols": numeric_cols,
    "label_cols": label_cols
}

with open("encoders_scalers.pkl", "wb") as f:
    pickle.dump(artifacts, f)
print("💾 Saved: encoders_scalers.pkl")

#---------------------------------------------------------------------------------------------------


# ============================================
# anime_test.csv → meta_preprocessed_test.csv 
# ============================================

import pandas as pd
import numpy as np
import pickle

# -----------------------------
# 1️⃣ 테스트 데이터 로드
# -----------------------------
meta_test = pd.read_csv("anime_test.csv")
print(f"✅ Loaded anime_test.csv, shape: {meta_test.shape}")

# 컬럼명 통일
if "anime_id" in meta_test.columns:
    meta_test.rename(columns={"anime_id": "MAL_ID"}, inplace=True)

# 결측 처리
meta_test = meta_test.fillna({
    'genre': '',
    'type': 'Unknown'
})

# ID 정수형 변환
meta_test['MAL_ID'] = meta_test['MAL_ID'].astype(int)

# -----------------------------
# 2️⃣ train에서 학습된 TF-IDF / Scaler 불러오기
# -----------------------------
with open("encoders_scalers.pkl", "rb") as f:
    artifacts = pickle.load(f)

tfidf_genre = artifacts["tfidf_genre"]
scaler = artifacts["scaler"]

# -----------------------------
# 3️⃣ TF-IDF 변환 (train vocab 기반)
# -----------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

# train과 컬럼명이 다르므로 genre 컬럼만 매핑
mat = tfidf_genre.transform(meta_test['genre'])
tfidf_df = pd.DataFrame(
    mat.toarray(),
    columns=[f"Genre_{t}" for t in tfidf_genre.get_feature_names_out()]
)

# -----------------------------
# 4️⃣ Numeric Scaling (train 스케일러 일부 사용)
# -----------------------------
# train numeric_cols에서 교집합만 사용
numeric_cols = ['rating', 'episodes', 'members']
for col in numeric_cols:
    meta_test[col] = pd.to_numeric(meta_test[col], errors='coerce')
meta_test[numeric_cols] = meta_test[numeric_cols].fillna(meta_test[numeric_cols].mean())

# 새로운 scaler 하나 더 만들어 적용 (독립적 테스트셋이므로)
from sklearn.preprocessing import MinMaxScaler
scaler_test = MinMaxScaler()
meta_test[numeric_cols] = scaler_test.fit_transform(meta_test[numeric_cols])

# -----------------------------
# 5️⃣ Feature 결합 및 저장
# -----------------------------
meta_test_processed = pd.concat(
    [meta_test[['MAL_ID'] + numeric_cols], tfidf_df],
    axis=1
)

meta_test_processed.to_csv("meta_preprocessed_test.csv", index=False)
print("💾 Saved: meta_preprocessed_test.csv")




#-----------------------------------------------------------------------------------------------------
# ============================================
# Hybrid Recommender Evaluation (Test set: rating_test.csv + meta_preprocessed_test.csv)
# ============================================

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# -----------------------------
# 1️⃣ Train + Test Data Load
# -----------------------------
train_df = pd.read_csv("rating_complete.csv")
test_df = pd.read_csv("rating_test.csv")
meta_train = pd.read_csv("meta_preprocessed.csv")
meta_test = pd.read_csv("meta_preprocessed_test.csv")

train_df = train_df[train_df['rating'] > 0]
test_df = test_df[test_df['rating'] > 0]

print(f"✅ Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# -----------------------------
# 2️⃣ CF Model (SVD, train으로만 학습)
# -----------------------------
reader = Reader(rating_scale=(train_df['rating'].min(), train_df['rating'].max()))
data = Dataset.load_from_df(train_df[['user_id', 'anime_id', 'rating']], reader)
trainset = data.build_full_trainset()

svd = SVD(n_factors=100, n_epochs=15, random_state=42, verbose=True)
svd.fit(trainset)

# -----------------------------
# 3️⃣ Content Matrix (train+test merge)
# -----------------------------
meta_all = pd.concat([meta_train, meta_test], ignore_index=True).drop_duplicates("MAL_ID")
# 🔹 NaN 값 전부 0으로 대체 (CB 유사도 계산 시 안전)
meta_all = meta_all.fillna(0.0)

feature_cols = [c for c in meta_all.columns if c != 'MAL_ID']
scaler = StandardScaler()
X = scaler.fit_transform(meta_all[feature_cols].values)
malid_to_idx = {aid: i for i, aid in enumerate(meta_all['MAL_ID'].values)}

# -----------------------------
# 4️⃣ Scoring Functions
# -----------------------------
def cf_score(user_id, anime_id):
    try:
        return svd.predict(user_id, anime_id).est
    except:
        return 0.0

def cb_score(user_id, anime_id, like_th=7.0):
    if anime_id not in malid_to_idx:
        return 0.0
    user_hist = train_df[train_df['user_id'] == user_id]
    liked = user_hist[user_hist['rating'] >= like_th]['anime_id'].values
    liked_idxs = [malid_to_idx[a] for a in liked if a in malid_to_idx][:30]
    if len(liked_idxs) == 0:
        return 0.0

    v = X[malid_to_idx[anime_id]].reshape(1, -1)
    L = X[liked_idxs]
    sims = cosine_similarity(L, v).flatten()
    return float(np.mean(sims))

def hybrid_score(user_id, anime_id, alpha=0.7, like_th=7.0):
    cf = cf_score(user_id, anime_id)
    cb = cb_score(user_id, anime_id, like_th)
    return alpha * cf + (1 - alpha) * cb

# -----------------------------
# 5️⃣ Precision / Recall (Test set)
# -----------------------------
def precision_recall_fast(user_id, scoring_fn, k=10, like_th=7.0, sample_items=3000):
    user_test = test_df[test_df['user_id'] == user_id]
    actual_liked = set(user_test[user_test['rating'] >= like_th]['anime_id'])
    if len(actual_liked) == 0:
        return None, None

    rated_items = set(train_df.loc[train_df['user_id'] == user_id, 'anime_id'])
    all_items = [a for a in meta_all['MAL_ID'].values if a not in rated_items]

    if len(all_items) > sample_items:
        np.random.seed(42)
        all_items = np.random.choice(all_items, sample_items, replace=False)

    scores = [(aid, scoring_fn(user_id, aid)) for aid in all_items]
    topk = [aid for aid, s in sorted(scores, key=lambda x: x[1], reverse=True)[:k]]
    hits = len(set(topk) & actual_liked)
    return hits / k, hits / len(actual_liked)

# -----------------------------
# 6️⃣ Evaluation Wrapper
# -----------------------------
def evaluate_fast(model_name, scoring_fn, user_sample=15, k_values=[5, 10]):
    valid_users = train_df.groupby('user_id').size()
    active_users = valid_users[valid_users >= 10].index
    users = test_df[test_df['user_id'].isin(active_users)]['user_id'].drop_duplicates().sample(user_sample, random_state=42)

    results = []
    for k in k_values:
        precisions, recalls = [], []
        for u in tqdm(users, desc=f"{model_name} (Top-{k})"):
            p, r = precision_recall_fast(u, scoring_fn, k=k)
            if p is not None:
                precisions.append(p)
                recalls.append(r)
        results.append((model_name, k, np.mean(precisions), np.mean(recalls)))
        print(f"\n📊 {model_name} (Top-{k}) → P@{k}: {np.mean(precisions):.4f}, R@{k}: {np.mean(recalls):.4f}")

    return results

# -----------------------------
# 7️⃣ Evaluate CF / CB / Hybrid (α-grid)
# -----------------------------
alphas = [0.3, 0.5, 0.7, 0.9]
results = []

# CF-only
results += evaluate_fast("CF-only", cf_score, user_sample=15, k_values=[5, 10])

# CB-only
results += evaluate_fast("CB-only", cb_score, user_sample=15, k_values=[5, 10])

# Hybrid α-grid
for a in alphas:
    fn = lambda u, aid, alpha=a: hybrid_score(u, aid, alpha=alpha)
    results += evaluate_fast(f"Hybrid (α={a})", fn, user_sample=15, k_values=[5, 10])

# -----------------------------
# 8️⃣ 결과 정리
# -----------------------------
res_df = pd.DataFrame(results, columns=["Model", "Top-K", "Precision", "Recall"])
print("\n📊 Model Comparison (Test Set, Top-5 & Top-10):")
print(res_df)
