import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
from autoint import AutoIntModel, predict_model

@st.cache_resource
def load_data():
    """
    앱에서 보여줄 필요 데이터를 가져오는 함수입니다.
    """
    project_path = os.path.abspath(os.getcwd())
    data_dir_nm = 'data'
    movielens_dir_nm = 'ml-1m'
    model_dir_nm = 'model'
    data_path = os.path.join(project_path, data_dir_nm)
    model_path = os.path.join(project_path, model_dir_nm)
    
    # 데이터 로드
    ratings_df = pd.read_csv(os.path.join(data_path, movielens_dir_nm, 'ratings_prepro.csv'))
    movies_df = pd.read_csv(os.path.join(data_path, movielens_dir_nm, 'movies_prepro.csv'))
    user_df = pd.read_csv(os.path.join(data_path, movielens_dir_nm, 'users_prepro.csv'))
    label_encoders = joblib.load(os.path.join(data_path, 'label_encoders.pkl'))
    field_dims = np.load(os.path.join(data_path, 'field_dims.npy'))

    # 모델 구조 생성
    dropout = 0.4
    embed_dim = 16
    model = AutoIntModel(field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True,
                         l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=dropout, init_std=0.0001)

    # 모델의 가중치를 불러오기 전에, 입력 텐서의 shape을 맞추기 위해 한번 호출해줍니다.
    model(tf.constant([[0] * len(field_dims)], dtype=tf.int64))

    # 이름(by_name)을 기준으로 가중치를 불러옵니다.
    weights_path = os.path.join(model_path, 'autoInt_model_weights.weights.h5')
    model.load_weights(weights_path, by_name=True)
    
    st.success("데이터 및 모델 로딩 완료!")
    
    return user_df, movies_df, ratings_df, model, label_encoders

# (이하 코드는 동일)
def get_user_seen_movies(ratings_df):
    """
    사용자가 과거에 보았던 영화 리스트를 가져옵니다.
    """
    return ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()

def get_user_non_seen_dict(movies_df, user_df, user_seen_movies):
    """
    사용자가 보지 않았던 영화 리스트를 가져옵니다.
    """
    unique_movies = movies_df['movie_id'].unique()
    unique_users = user_df['user_id'].unique()
    user_non_seen_dict = {}

    for user in unique_users:
        user_seen_movie_list = user_seen_movies[user_seen_movies['user_id'] == user]['movie_id'].values
        if len(user_seen_movie_list) > 0:
            user_seen_movie_list = user_seen_movie_list[0]
        else:
            user_seen_movie_list = []
        user_non_seen_movie_list = list(set(unique_movies) - set(user_seen_movie_list))
        user_non_seen_dict[user] = user_non_seen_movie_list
        
    return user_non_seen_dict


def get_user_info(user_id, users_df):
    """
    사용자 정보를 가져옵니다.
    """
    return users_df[users_df['user_id'] == user_id]

def get_user_past_interactions(user_id, ratings_df, movies_df):
    """
    사용자 평점 데이터 중 4점 이상(선호했다는 정보)만 가져옵니다. 
    """
    return ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)].merge(movies_df, on='movie_id')


def get_recom(user, user_non_seen_dict, user_df, movies_df, r_year, r_month, model, label_encoders):
    """
    아래와 같은 순서로 추천 결과를 가져옵니다.
    1. streamlit에서 입력 받은 타겟 월, 연도, 사용자 정보를 받아옴
    2. 사용자가 보지 않았던 정보 추출
    3. model input으로 넣을 수 있는 형태로 데이터프레임 구성
    4. label encoder 적용해 모델에 넣을 준비
    5. 모델 predict 수행
    """
    user_non_seen_movie = user_non_seen_dict.get(user, [])
    if not user_non_seen_movie:
        return pd.DataFrame() # 보지 않은 영화가 없으면 빈 데이터프레임 반환

    user_id_list = [user for _ in range(len(user_non_seen_movie))]
    r_decade = str(r_year - (r_year % 10)) + 's'
    
    user_non_seen_movie_df = pd.merge(pd.DataFrame({'movie_id': user_non_seen_movie}), movies_df, on='movie_id')
    user_info_df = pd.merge(pd.DataFrame({'user_id': user_id_list}), user_df, on='user_id')
    user_info_df['rating_year'] = r_year
    user_info_df['rating_month'] = r_month
    user_info_df['rating_decade'] = r_decade
    
    merge_data = pd.concat([user_non_seen_movie_df.reset_index(drop=True), user_info_df.reset_index(drop=True)], axis=1)
    merge_data.fillna('no', inplace=True)
    
    feature_cols = ['user_id', 'movie_id','movie_decade', 'movie_year', 'rating_year', 'rating_month', 'rating_decade', 'genre1','genre2', 'genre3', 'gender', 'age', 'occupation', 'zip']
    merge_data = merge_data[feature_cols]
    
    for col, le in label_encoders.items():
        # 학습된 인코더에 알려진 모든 레이블을 가져옵니다.
        known_classes = set(le.classes_)
        # 기본값으로 첫 번째 클래스를 사용합니다.
        default_value = list(known_classes)[0]
        # 알려지지 않은 레이블을 기본값으로 대체합니다.
        merge_data[col] = merge_data[col].apply(lambda x: x if x in known_classes else default_value)
        # 이제 안전하게 변환을 수행합니다.
        merge_data[col] = le.transform(merge_data[col])
    
    # 모델 내부의 타입(int64)과 입력 데이터의 타입(int32) 불일치 에러를 해결하기 위해
    # 입력 데이터의 타입을 int64로 변환합니다.
    merge_data_int64 = merge_data.astype(np.int64)

    # --- 🕵️‍♂️ 디버깅 코드 ---
    # 모델의 원시 예측 결과를 직접 확인합니다.
    raw_predictions = model.predict(merge_data_int64)
    
    # 예측 결과와 영화 ID를 합쳐서 데이터프레임으로 만듭니다.
    pred_df = pd.DataFrame({
        'movie_id': merge_data['movie_id'], # 인코딩 전의 movie_id 사용
        'prediction_score': raw_predictions.flatten()
    })

    # 예측 점수를 내림차순으로 정렬하여 상위 20개를 확인합니다.
    st.write("---")
    st.write("##### 🕵️‍♂️ 모델 예측 점수 확인 (상위 20개)")
    st.dataframe(pred_df.sort_values(by='prediction_score', ascending=False).head(20))
    # --- 디버깅 코드 끝 ---


    recom_top = predict_model(model, merge_data_int64)
    if not recom_top:
        return pd.DataFrame()

    recom_top = [r[0] for r in recom_top]
    origin_m_id = label_encoders['movie_id'].inverse_transform(recom_top)
    
    return movies_df[movies_df['movie_id'].isin(origin_m_id)]

# --- Streamlit UI 구성 ---

st.title("🎬 영화 추천 시스템")

# 데이터 로드
try:
    users_df, movies_df, ratings_df, model, label_encoders = load_data()
    user_seen_movies = get_user_seen_movies(ratings_df)
    user_non_seen_dict = get_user_non_seen_dict(movies_df, users_df, user_seen_movies)

    st.header("👤 사용자 정보를 넣어주세요.")
    user_id = st.number_input("사용자 ID 입력", min_value=users_df['user_id'].min(), max_value=users_df['user_id'].max(), value=users_df['user_id'].min())
    r_year = st.number_input("추천 타겟 연도 입력", min_value=ratings_df['rating_year'].min(), max_value=ratings_df['rating_year'].max(), value=ratings_df['rating_year'].min())
    r_month = st.number_input("추천 타겟 월 입력", min_value=ratings_df['rating_month'].min(), max_value=ratings_df['rating_month'].max(), value=ratings_df['rating_month'].min())

    if st.button("추천 결과 보기"):
        st.write("---")
        st.subheader(f"👤 사용자 ID: {user_id} 님")
        
        st.write("##### 📋 사용자 기본 정보")
        user_info = get_user_info(user_id, users_df)
        st.dataframe(user_info)

        st.write("##### 👍 과거에 선호했던 영화 (평점 4점 이상)")
        user_interactions = get_user_past_interactions(user_id, ratings_df, movies_df)
        st.dataframe(user_interactions)

        st.write("---")
        st.subheader("✨ 추천 결과")
        with st.spinner('추천 영화를 가져오는 중입니다...'):
            recommendations = get_recom(user_id, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, label_encoders)
            
            if recommendations.empty:
                st.warning("추천 영화를 찾지 못했습니다. 모델이 모든 영화에 대해 낮은 점수를 예측했거나, 다른 사용자가 이미 모든 영화를 평가했을 수 있습니다.")
            else:
                st.dataframe(recommendations)
except Exception as e:
    st.error(f"앱 실행 중 에러가 발생했습니다: {e}")
