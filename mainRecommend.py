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


#--------------------------------------------------------------------------



# ============================================
# Hybrid Recommender Evaluation (CF / CB / Hybrid α-grid)             추천
# ============================================

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# -----------------------------
# 1️⃣ Data Load
# -----------------------------
rating_df = pd.read_csv("rating_complete.csv")
meta_df = pd.read_csv("meta_preprocessed.csv")

rating_df = rating_df[rating_df['rating'] > 0]
train_df, test_df = train_test_split(rating_df, test_size=0.2, random_state=42)
print(f"✅ Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# -----------------------------
# 2️⃣ CF Model (SVD)
# -----------------------------
reader = Reader(rating_scale=(rating_df['rating'].min(), rating_df['rating'].max()))
data = Dataset.load_from_df(train_df[['user_id', 'anime_id', 'rating']], reader)
trainset = data.build_full_trainset()

svd = SVD(n_factors=100, n_epochs=15, random_state=42, verbose=True)
svd.fit(trainset)

# -----------------------------
# 3️⃣ Content Matrix
# -----------------------------
feature_cols = [c for c in meta_df.columns if c != 'MAL_ID']
scaler = StandardScaler()
X = scaler.fit_transform(meta_df[feature_cols].values)
malid_to_idx = {aid: i for i, aid in enumerate(meta_df['MAL_ID'].values)}

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
    liked_idxs = [malid_to_idx[a] for a in liked if a in malid_to_idx][:30]  # top30으로 완화
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
# 5️⃣ Precision / Recall
# -----------------------------
def precision_recall_fast(user_id, scoring_fn, k=10, like_th=7.0, sample_items=3000):
    user_test = test_df[test_df['user_id'] == user_id]
    actual_liked = set(user_test[user_test['rating'] >= like_th]['anime_id'])
    if len(actual_liked) == 0:
        return None, None

    rated_items = set(train_df.loc[train_df['user_id'] == user_id, 'anime_id'])
    all_items = [a for a in meta_df['MAL_ID'].values if a not in rated_items]

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
# ⭐ Re-ranking 기반 추천 함수 (CF 후보 → CB로 재정렬)
# -----------------------------
def recommend_rerank(user_id, k=10, N=200, like_th=7.0, alpha=0.7):
    # 1) CF로 넓게 후보 N개 뽑기
    rated = set(train_df.loc[train_df['user_id']==user_id, 'anime_id'])
    candidates = [a for a in meta_df['MAL_ID'].values if a not in rated]

    cf_scores = [(aid, cf_score(user_id, aid)) for aid in candidates]
    topN = sorted(cf_scores, key=lambda x: x[1], reverse=True)[:N]
    
    # 2) 후보 N개에만 CB 계산 후 하이브리드 재정렬
    rescored = []
    for aid, cf in topN:
        cb = cb_score(user_id, aid, like_th)
        hy = alpha * cf + (1 - alpha) * cb
        rescored.append((aid, hy))
    
    # 최종 Top-K 추출
    topk = [aid for aid, _ in sorted(rescored, key=lambda x: x[1], reverse=True)[:k]]
    return topk


# 🎯 유저 샘플 1명
user_id = train_df['user_id'].sample(1, random_state=42).values[0]
recommendations = recommend_rerank(user_id, k=10, N=200, alpha=0.7)

# 🔍 추천된 anime_id 기반 상세정보 매칭
anime_info = pd.read_csv("anime.csv")[["MAL_ID", "Name", "Genres", "Score", "Type", "Episodes"]]
rec_details = anime_info[anime_info["MAL_ID"].isin(recommendations)]

print(f"\n🎯 추천 결과 (User {user_id}) — α=0.5")
for _, row in rec_details.iterrows():
    print(f"- {row['Name']} | 🎬 {row['Type']} | ⭐ {row['Score']} | 🧩 {row['Genres'][:60]}...")
