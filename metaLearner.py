# ======================================================
# ⚡ CF(SVD) + CB(Content-Based) Hybrid Meta-Train Builder (Fast Version)
# ======================================================

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# ------------------------------------------------------
# 1️⃣ 데이터 로드
# ------------------------------------------------------
meta = pd.read_csv("meta_preprocessed.csv")
ratings_train = pd.read_csv("rating_complete.csv")
ratings_test = pd.read_csv("rating_test.csv")

meta = meta.fillna(0.0)
ratings_train = ratings_train[ratings_train['rating'] > 0]
ratings_test = ratings_test[ratings_test['rating'] > 0]

print(f"✅ Train: {ratings_train.shape}, Test: {ratings_test.shape}\n")

# ------------------------------------------------------
# 2️⃣ CF 모델 (SVD)
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
print("🎯 CF (SVD) 학습 완료!\n")

# ------------------------------------------------------
# 3️⃣ 콘텐츠 피처 스케일링 및 유사도 행렬 계산
# ------------------------------------------------------
feature_cols = [c for c in meta.columns if c != 'MAL_ID']
scaler = StandardScaler()
X = scaler.fit_transform(meta[feature_cols].values)

malid_to_idx = {aid: i for i, aid in enumerate(meta['MAL_ID'].values)}
idx_to_malid = {v: k for k, v in malid_to_idx.items()}

print("🧮 전체 아이템 유사도 행렬 계산 중...")
item_sim_matrix = cosine_similarity(X)
print("✅ 유사도 행렬 계산 완료!\n")

# ------------------------------------------------------
# 4️⃣ CF 점수 함수
# ------------------------------------------------------
def cf_score(user_id, anime_id):
    try:
        return svd.predict(user_id, anime_id).est
    except:
        return np.nan

# ------------------------------------------------------
# 5️⃣ CB 점수 벡터화 버전 (유저 단위 캐싱)
# ------------------------------------------------------
def get_cb_scores_for_user(user_id, like_th=8.0):
    """유저가 좋아한 아이템들의 평균 유사도 벡터를 미리 계산해둔다"""
    user_train = ratings_train[ratings_train['user_id'] == user_id]
    liked = user_train[user_train['rating'] >= like_th]['anime_id'].values
    liked_idx = [malid_to_idx[a] for a in liked if a in malid_to_idx]
    if not liked_idx:
        return None  # 선호 아이템 없음
    # 좋아한 애니들의 유사도 평균 (아이템 전체에 대해)
    return item_sim_matrix[liked_idx].mean(axis=0)

# ------------------------------------------------------
# 6️⃣ 메타트레인 생성 (1000명 샘플링)
# ------------------------------------------------------
np.random.seed(42)
all_users = ratings_train['user_id'].unique()
sample_users = np.random.choice(all_users, size=min(1000, len(all_users)), replace=False)

meta_train_records = []

print(f"👥 총 {len(all_users)}명 중 {len(sample_users)}명 유저 샘플링하여 CB 계산\n")

for uid in tqdm(sample_users, desc="Building Fast Meta-Train"):
    user_data = ratings_train[ratings_train['user_id'] == uid]
    cb_vector = get_cb_scores_for_user(uid)
    if cb_vector is None:
        continue
    for row in user_data.itertuples(index=False):
        aid = row.anime_id
        if aid not in malid_to_idx:
            continue
        cb = cb_vector[malid_to_idx[aid]]  # 캐시된 유사도에서 바로 꺼냄|
        cf = cf_score(uid, aid)
        meta_train_records.append((uid, aid, cf, cb, row.rating))

meta_train_df = pd.DataFrame(meta_train_records, columns=["user_id", "anime_id", "cf_score", "cb_score", "true_rating"])

# ------------------------------------------------------
# 7️⃣ 결과 저장
# ------------------------------------------------------
meta_train_df.to_csv("meta_train_ready.csv", index=False)
print("\n💾 meta_train_ready.csv 저장 완료!")
print(meta_train_df.head())

# ✅ 최종 출력 예시
print(f"\n📊 Meta-Train 구성 완료: {meta_train_df.shape[0]} samples")
print(meta_train_df.describe())


# ======================================================
# 모델로 추천
# ======================================================


# ------------------------------------------------------
# 1️⃣ 데이터 로드
# ------------------------------------------------------
meta_df = pd.read_csv("meta_train_ready.csv")
anime_info = pd.read_csv("anime.csv")  # 애니 이름 매칭용

print(f"✅ Meta-Train Data: {meta_df.shape}")
print(f"✅ Anime Info Data: {anime_info.shape}\n")

# ------------------------------------------------------
# 2️⃣ 입력 / 타깃 분리
# ------------------------------------------------------
X = meta_df[["cf_score", "cb_score"]]
y = meta_df["true_rating"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape}, Validation: {X_val.shape}\n")

# ------------------------------------------------------
# 3️⃣ 메타러너 (XGBoost)
# ------------------------------------------------------
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    objective="reg:squarederror"
)

print("🎯 Training Meta-Learner (XGBoost)...")
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("✅ Meta-Learner Training Complete!\n")

# ------------------------------------------------------
# 4️⃣ RMSE 평가
# ------------------------------------------------------
val_pred = xgb_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"📉 Validation RMSE: {rmse:.4f}\n")

# ------------------------------------------------------
# 5️⃣ 유저별 추천 Top-10 (이름 포함)
# ------------------------------------------------------
sample_user = meta_df["user_id"].sample(1, random_state=42).iloc[0]
user_data = meta_df[meta_df["user_id"] == sample_user].copy()

# 예측 평점 계산
user_data["meta_pred"] = xgb_model.predict(user_data[["cf_score", "cb_score"]])

# 애니 이름 병합
user_data = user_data.merge(anime_info[["MAL_ID", "Name"]], left_on="anime_id", right_on="MAL_ID", how="left")

# 상위 10개 추천
top10 = user_data.sort_values("meta_pred", ascending=False).head(10)

print(f"🎯 User {sample_user}의 추천 Top-10 애니:")
for _, row in top10.iterrows():
    print(f" - {row['Name']:<40} | 예측평점: {row['meta_pred']:.2f} | 실제: {row['true_rating']:.1f}")

# ------------------------------------------------------
# 6️⃣ 모델 저장 (선택)
# ------------------------------------------------------
xgb_model.save_model("meta_learner_xgb.json")
print("\n💾 Meta-Learner 모델 저장 완료!")
