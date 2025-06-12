import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

def preprocess_earthquake_data(df):
    """
    Preprocesses the earthquake DataFrame by converting 'Magnitude' to numeric,
    dropping NaNs, and rounding 'Magnitude' to one decimal place.
    """
    df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
    df = df.dropna(subset=['Magnitude'])
    df['Magnitude_Rounded'] = df['Magnitude'].round(1)
    return df

def count_and_log_transform(df):
    """
    Counts occurrences for each rounded magnitude and applies a log10 transformation.
    """
    mag_counts = df['Magnitude_Rounded'].value_counts().sort_index()
    # Ensure no log of zero if a magnitude has no counts, although value_counts handles this
    # Filter out magnitudes with zero counts before taking log10
    mag_counts_filtered = mag_counts[mag_counts > 0]
    log_counts = np.log10(mag_counts_filtered)
    return mag_counts, log_counts

def fit_gutenberg_richter(X, y):
    """
    Fits the Gutenberg-Richter law using basic linear regression with numpy.
    Calculates a, b, predicted values, and residuals.
    """
    # Add a bias (intercept) term to X
    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    # Calculate coefficients using the normal equation: theta = (X_b.T * X_b)^-1 * X_b.T * y
    # Using np.linalg.lstsq for numerical stability and handling singular matrices
    coefficients, residuals_sum_squares, rank, s = np.linalg.lstsq(X_b, y, rcond=None)

    a = coefficients[0]  # Intercept
    b = -coefficients[1] # Negative of the slope

    y_pred = X_b @ coefficients # Predict y values

    # Calculate residuals manually: observed - predicted
    residuals = y.flatten() - y_pred.flatten()

    return a, b, y_pred.flatten(), residuals

def make_result_df(mag_counts, log_counts, y_pred, residuals):
    """
    Creates a DataFrame summarizing observed counts, log counts, predicted log counts, and residuals.
    Aligns DataFrames based on magnitude index.
    """
    # Ensure mag_counts and log_counts are aligned by index
    combined_df = pd.DataFrame({
        'Count': mag_counts,
        'log10(Count)': log_counts
    }).reset_index().rename(columns={'index': 'Magnitude'})

    # Create a DataFrame for predictions and residuals, ensuring alignment
    # Use the 'Magnitude' column from the combined_df to ensure consistent index
    pred_res_df = pd.DataFrame({
        'Magnitude': combined_df['Magnitude'],
        'Predicted log10(Count)': y_pred,
        'Residual': residuals
    })

    # Merge the two DataFrames on 'Magnitude' to ensure all data is correctly aligned
    result_df = pd.merge(combined_df, pred_res_df, on='Magnitude', how='inner')
    return result_df

def plot_bar_counts(result_df):
    """
    Generates a bar plot of observed earthquake counts per magnitude.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=result_df['Magnitude'],
        y=result_df['Count'],
        marker_color='skyblue',
        name='Observed Count'
    ))
    fig.update_layout(
        title="규모별 지진 발생 횟수",
        xaxis_title="Magnitude",
        yaxis_title="Occurrences"
    )
    return fig

def plot_regression(result_df, a, b):
    """
    Generates a scatter plot of observed log10(counts) and a line plot
    of the predicted Gutenberg-Richter law.
    """
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
        title=f"log$_{{10}}$(N) = {a:.2f} - {b:.2f}M",
        xaxis_title="Magnitude",
        yaxis_title="log$_{{10}}$(Occurrences)"
    )
    return fig

def plot_residuals(result_df):
    """
    Generates a bar plot of the residuals (observed - predicted log10(counts)).
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=result_df['Magnitude'],
        y=result_df['Residual'],
        marker_color='orange',
        name='Residual'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title="잔차 (관측값 - 예측값)",
        xaxis_title="Magnitude",
        yaxis_title="Residual"
    )
    return fig

# --- Streamlit Application ---
st.set_page_config(layout="wide")
st.title("지진 구텐베르크-릭터 법칙 분석")

st.write("지진 데이터셋을 업로드하여 구텐베르크-릭터 법칙을 분석하고 시각화합니다. Magnitude(규모) 열이 포함된 CSV 파일을 사용해주세요.")

uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("원본 데이터 미리보기")
    st.dataframe(df.head())

    # 1. Preprocess the data
    processed_df = preprocess_earthquake_data(df.copy()) # Use .copy() to avoid SettingWithCopyWarning

    st.subheader("전처리된 데이터 요약")
    st.write(f"총 유효 지진 기록: {len(processed_df)}개")
    st.dataframe(processed_df.head())

    # 2. Count and log transform
    mag_counts, log_counts = count_and_log_transform(processed_df)

    # Filter out magnitudes with zero counts for regression
    # This ensures X and y have consistent lengths
    valid_magnitudes = log_counts.index.values
    valid_log_counts = log_counts.values

    X = valid_magnitudes.reshape(-1, 1) # Magnitude
    y = valid_log_counts.reshape(-1, 1) # log10(Count)

    if len(X) > 1: # Ensure enough data points for regression
        # 3. Fit Gutenberg-Richter law
        a, b, y_pred, residuals = fit_gutenberg_richter(X, y)

        st.subheader("구텐베르크-릭터 법칙 파라미터")
        st.info(f"$\\log_{{10}}(N) = \\mathbf{{{a:.2f}}} - \\mathbf{{{b:.2f}}}M$")
        st.markdown(f"- $a$ 값 (절편): {a:.2f}")
        st.markdown(f"- $b$ 값 (기울기): {b:.2f}")
        st.markdown("($N$: 규모 $M$ 이상의 지진 횟수)")

        # 4. Make result DataFrame
        result_df = make_result_df(mag_counts, log_counts, y_pred, residuals)

        st.subheader("분석 결과 데이터프레임")
        st.dataframe(result_df)

        # 5. Plotting
        st.subheader("시각화")

        col1, col2 = st.columns(2)

        with col1:
            fig_bar = plot_bar_counts(result_df)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_regression = plot_regression(result_df, a, b)
            st.plotly_chart(fig_regression, use_container_width=True)

        st.subheader("잔차 분석")
        fig_residuals = plot_residuals(result_df)
        st.plotly_chart(fig_residuals, use_container_width=True)

        st.write("잔차는 관측값과 예측값의 차이를 나타냅니다. 잔차가 0에 가까울수록 모델이 데이터를 잘 설명한다고 볼 수 있습니다.")

    else:
        st.warning("회귀 분석을 수행하기에 충분한(2개 이상의) 유효한 규모 데이터 포인트가 없습니다.")

else:
    st.info("시작하려면 CSV 파일을 업로드해주세요.")
