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
# 7️⃣ 저장 
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
