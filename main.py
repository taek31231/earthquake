import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from sklearn.linear_model import LinearRegression
from utils import (
    preprocess_earthquake_data,
    count_and_log_transform,
    fit_gutenberg_richter,
    make_result_df,
    plot_bar_counts,
    plot_regression,
    plot_residuals
)

# 페이지 설정
st.set_page_config(page_title="지진 발생 분석 대시보드", layout="wide")
st.title("🌍 전 세계 지진 분석 (2001~2023)")

# 데이터 업로드
df = pd.read_csv("data/database.csv")
df = preprocess_earthquake_data(df)

# 국가 필터링을 위한 좌표 범위 설정
country_bounds = {
    "Japan": [30, 46, 129, 146],
    "USA": [24, 50, -125, -66],
    "Chile": [-56, -17, -76, -66],
    "Indonesia": [-11, 6, 95, 141],
    "South Korea": [33, 39, 124, 130.5],
    "Turkey": [36, 42, 26, 45],
    "Iceland": [63, 67, -25, -13],
    "Mid-Atlantic Ridge": [-60, 60, -40, -10],
    "New Zealand": [-47, -33, 165, 180],
    "Philippines": [5, 20, 117, 127]
}


selected_country = st.selectbox("국가 선택", list(country_bounds.keys()))
lat_min, lat_max, lon_min, lon_max = country_bounds[selected_country]

# 선택한 국가의 지진만 필터링 (규모 3.0 ~ 7.5)
df_filtered = df[
    (df['Latitude'].between(lat_min, lat_max)) &
    (df['Longitude'].between(lon_min, lon_max)) &
    (df['Magnitude'] >= 3.0) &
    (df['Magnitude'] <= 7.5)
]

st.markdown(f"### 🇨🇭 {selected_country}에서 발생한 지진 데이터 수: {len(df_filtered)}개")

# 지진 지도 시각화
st.subheader("🗺️ 지진 발생 지점 (Folium 지도)")
mid_lat = (lat_min + lat_max) / 2
mid_lon = (lon_min + lon_max) / 2

m = folium.Map(location=[mid_lat, mid_lon], zoom_start=5)

for _, row in df_filtered.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=2 + (row['Magnitude'] - 3) * 2,
        color='red',
        fill=True,
        fill_opacity=0.7,
        popup=f"Magnitude: {row['Magnitude']}"
    ).add_to(m)

folium_static(m)

# Gutenberg-Richter 분석
st.subheader("📈 Gutenberg-Richter 법칙 분석")
mag_counts, log_counts = count_and_log_transform(df_filtered)
X = mag_counts.index.values.reshape(-1, 1)
y = log_counts.values.reshape(-1, 1)

a, b, y_pred, residuals = fit_gutenberg_richter(X, y)
result_df = make_result_df(mag_counts, log_counts, y_pred, residuals)

# 시각화
st.plotly_chart(plot_bar_counts(result_df), use_container_width=True)
st.plotly_chart(plot_regression(result_df, a, b), use_container_width=True)
st.plotly_chart(plot_residuals(result_df), use_container_width=True)

# 가장 잔차가 큰 지진 규모 표시
worst_residual = result_df.loc[result_df['Residual'].idxmax()]
st.markdown(f"#### 📌 곧 발생 가능성이 가장 높은 지진의 규모는: **{worst_residual['Magnitude']}**")
