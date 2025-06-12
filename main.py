import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
    log_counts = np.log10(mag_counts[mag_counts > 0])
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
    # lstsq returns (coefficients, residuals, rank, singular_values)
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
    pred_res_df = pd.DataFrame({
        'Magnitude': mag_counts.index, # Use the same index as mag_counts
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
        title=f"log$_{{10}}$(N) = {a:.2f} - {b:.2f}M", # LaTeX-like formatting for title
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

# Example Usage (assuming you have a DataFrame 'df' loaded)
if __name__ == '__main__':
    # Create a dummy DataFrame for demonstration
    data = {
        'Magnitude': [1.2, 1.3, 1.5, 1.5, 1.8, 2.0, 2.1, 2.1, 2.3, 2.5,
                      2.5, 2.6, 2.8, 3.0, 3.1, 3.2, 3.5, 3.5, 3.8, 4.0,
                      4.1, 4.2, 4.5, 4.8, 5.0, 5.1, 5.5, 5.8, 6.0, 6.2,
                      6.5, 6.8, 7.0]
    }
    df = pd.DataFrame(data)

    # 1. Preprocess the data
    processed_df = preprocess_earthquake_data(df.copy()) # Use .copy() to avoid SettingWithCopyWarning

    # 2. Count and log transform
    mag_counts, log_counts = count_and_log_transform(processed_df)

    # Prepare data for regression
    X = mag_counts.index.values.reshape(-1, 1) # Magnitude
    y = log_counts.values.reshape(-1, 1) # log10(Count)

    # 3. Fit Gutenberg-Richter law
    a, b, y_pred, residuals = fit_gutenberg_richter(X, y)

    print(f"Gutenberg-Richter Law: log10(N) = {a:.2f} - {b:.2f}M")

    # 4. Make result DataFrame
    result_df = make_result_df(mag_counts, log_counts, y_pred, residuals)
    print("\nResult DataFrame:")
    print(result_df)

    # 5. Plotting
    # Bar chart of counts
    fig_bar = plot_bar_counts(result_df)
    fig_bar.show()

    # Regression plot
    fig_regression = plot_regression(result_df, a, b)
    fig_regression.show()

    # Residuals plot
    fig_residuals = plot_residuals(result_df)
    fig_residuals.show()
