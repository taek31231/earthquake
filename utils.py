import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def preprocess_earthquake_data(df):
    df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
    df = df.dropna(subset=['Magnitude'])
    df['Magnitude_Rounded'] = df['Magnitude'].round(1)
    return df

def count_and_log_transform(df):
    mag_counts = df['Magnitude_Rounded'].value_counts().sort_index()
    log_counts = np.log10(mag_counts)
    return mag_counts, log_counts

def fit_gutenberg_richter(X, y):
    model = LinearRegression().fit(X, y)
    a = model.intercept_[0]
    b = -model.coef_[0][0]  # Gutenberg-Richter에서 b는 음수 부호로 표현
    y_pred = model.predict(X).flatten()
    residuals = y.flatten() - y_pred
    return a, b, y_pred, residuals

def make_result_df(mag_counts, log_counts, y_pred, residuals):
    return pd.DataFrame({
        'Magnitude': mag_counts.index,
        'Count': mag_counts.values,
        'log10(Count)': log_counts.values,
        'Predicted log10(Count)': y_pred,
        'Residual': residuals
    })

def plot_bar_counts(result_df):
    fig, ax = plt.subplots()
    ax.bar(result_df['Magnitude'], result_df['Count'], width=0.08, color='skyblue')
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Occurrences")
    ax.set_title("규모별 발생 횟수")
    return fig

def plot_regression(result_df, a, b):
    fig, ax = plt.subplots()
    ax.scatter(result_df['Magnitude'], result_df['log10(Count)'], label='Observed', color='blue')
    ax.plot(result_df['Magnitude'], result_df['Predicted log10(Count)'], label='Predicted (GR Law)', color='red')
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("log10(Occurrences)")
    ax.set_title(f"log10(N) = {a:.2f} - {b:.2f}M")
    ax.legend()
    return fig

def plot_residuals(result_df):
    fig, ax = plt.subplots()
    ax.bar(result_df['Magnitude'], result_df['Residual'], width=0.08, color='orange')
    ax.axhline(0, color='black', linestyle='--')
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Residual")
    ax.set_title("Anomaly (관측 log10 - 예측 log10)")
    return fig
