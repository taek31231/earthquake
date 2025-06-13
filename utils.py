import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

def preprocess_earthquake_data(df):
    df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
    df = df.dropna(subset=['Magnitude'])
    df = df[(df['Magnitude'] >= 3.0) & (df['Magnitude'] <= 7.5)]  # 필터 적용
    df['Magnitude_Rounded'] = df['Magnitude'].round(1)
    return df

def count_and_log_transform(df):
    mag_counts = df['Magnitude_Rounded'].value_counts().sort_index()
    log_counts = np.log10(mag_counts)
    return mag_counts, log_counts

def fit_gutenberg_richter(X, y):
    model = LinearRegression().fit(X, y)
    a = model.intercept_[0]
    b = -model.coef_[0][0]
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
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=result_df['Magnitude'],
        y=result_df['Count'],
        marker_color='skyblue',
        marker_line_color='black',
        marker_line_width=1
    ))
    fig.update_layout(
        title="규모별 발생 횟수",
        xaxis_title="Magnitude",
        yaxis_title="Occurrences",
        bargap=0.1
    )
    return fig

def plot_regression(result_df, a, b):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result_df['Magnitude'],
        y=result_df['log10(Count)'],
        mode='markers',
        name='Observed',
        marker=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=result_df['Magnitude'],
        y=result_df['Predicted log10(Count)'],
        mode='lines',
        name='Predicted (GR Law)',
        line=dict(color='red')
    ))
    fig.update_layout(
        title=f"log10(N) = {a:.2f} - {b:.2f}M",
        xaxis_title="Magnitude",
        yaxis_title="log10(Occurrences)"
    )
    return fig

def plot_residuals(result_df):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=result_df['Magnitude'],
        y=result_df['Residual'],
        marker_color='orange'
    ))
    fig.add_trace(go.Scatter(
        x=result_df['Magnitude'],
        y=[0]*len(result_df),
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Zero Line'
    ))
    fig.update_layout(
        title="Anomaly (관측 log10 - 예측 log10)",
        xaxis_title="Magnitude",
        yaxis_title="Residual"
    )
    return fig
