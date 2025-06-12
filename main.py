import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="지진 발생 가능성 분석", layout="wide")

st.title("🌍 Gutenberg-Richter 기반 지진 규모 예측")

# 파일 업로드
st.sidebar.header("CSV 데이터 업로드")
uploaded_file = st.sidebar.file_uploader("지진 데이터 (CSV)", type=["csv"])

if uploaded_file is not None:
    # 데이터 로드
    df = pd.read_csv(uploaded_file)
    df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
    df = df.dropna(subset=['Magnitude'])
    df['Magnitude_Rounded'] = df['Magnitude'].round(1)

    # 규모별 발생 횟수
    mag_counts = df['Magnitude_Rounded'].value_counts().sort_index()
    log_counts = np.log10(mag_counts)

    # 회귀 분석
    X = mag_counts.index.values.reshape(-1, 1)
    y = log_counts.values.reshape(-1, 1)

    model = LinearRegression().fit(X, y)
    a = model.intercept_[0]
    b = -model.coef_[0][0]

    y_pred = model.predict(X).flatten()
    residuals = log_counts.values - y_pred

    # 결과 데이터프레임
    result_df = pd.DataFrame({
        'Magnitude': mag_counts.index,
        'Count': mag_counts.values,
        'log10(Count)': log_counts.values,
        'Predicted log10(Count)': y_pred,
        'Residual': residuals
    })

    # 🔍 발생 가능성 높은 규모 추정 (예측보다 관측이 적은 규모 → anomaly가 큰 값)
    likely_mag = result_df.sort_values(by='Residual').iloc[-1]['Magnitude']

    # 📊 시각화
    st.subheader("1. 규모별 지진 발생 빈도 (실제)")
    fig1, ax1 = plt.subplots()
    ax1.bar(result_df['Magnitude'], result_df['Count'], width=0.08, color='skyblue')
    ax1.set_xlabel("Magnitude")
    ax1.set_ylabel("Occurrences")
    ax1.set_title("규모별 발생 횟수")
    st.pyplot(fig1)

    st.subheader("2. Gutenberg-Richter 회귀 분석")
    fig2, ax2 = plt.subplots()
    ax2.scatter(result_df['Magnitude'], result_df['log10(Count)'], label='Observed', color='blue')
    ax2.plot(result_df['Magnitude'], result_df['Predicted log10(Count)'], label='Predicted (GR Law)', color='red')
    ax2.set_xlabel("Magnitude")
    ax2.set_ylabel("log10(Occurrences)")
    ax2.set_title(f"log10(N) = {a:.2f} - {b:.2f}M")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("3. 잔차 (관측 - 예측) 분석")
    fig3, ax3 = plt.subplots()
    ax3.bar(result_df['Magnitude'], result_df['Residual'], width=0.08, color='orange')
    ax3.axhline(0, color='black', linestyle='--')
    ax3.set_xlabel("Magnitude")
    ax3.set_ylabel("Residual")
    ax3.set_title("잔차 (관측 log10 - 예측 log10)")
    st.pyplot(fig3)

    st.subheader("🔮 예측 결과")
    st.markdown(f"""
    - Gutenberg-Richter 회귀식:  
      \[
      \\log_{{10}}(N) = {a:.2f} - {b:.2f} \\cdot M
      \]
    - 관측 대비 예측보다 적게 발생한 규모 중 anomaly가 가장 큰 지진 규모는  
      **`{likely_mag:.1f}`** 이며, 앞으로 이 규모에서 지진이 발생할 가능성이 높습니다.
    """)

else:
    st.info("왼쪽 사이드바에서 지진 CSV 파일을 업로드하세요.")
