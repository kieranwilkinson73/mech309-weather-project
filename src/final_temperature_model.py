import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to find least squares solution
def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_aug = np.column_stack([np.ones(len(X)), X]) # Create an extra column of 1s to create the intercept term
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None) # Solve for the beta variables using least squares method in numpy linalg library
    return beta # Return the beta variables

# Use regression model to predict y
def predict(A: np.ndarray, x: np.ndarray):
    return A @ np.insert(x, 0, 1).reshape(-1, 1) # Calculate the predicted temperature given set of predictor variables

# Root Mean Square Error
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) # Return the root mean square error between predicted temp and validation temp

# Mean Absolute Error
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred))) # Return the mean absolute error between predicted temp and validation temp

if __name__ == "__main__":
    forecast_horizons = [1, 3, 6, 12, 24]

    feature_cols = [
        'T', 'W', 'sin_day', 'cos_day', 'sin_year', 'cos_year',
        'RH', 'P', 'Prec', 'Cloud', 'T_lag1', 'T_lag3', 'T_lag12'
    ]

    train_df = pd.read_csv("data/processed/train_weather.csv", index_col=0, parse_dates=True)
    val_df = pd.read_csv("data/processed/val_weather.csv", index_col=0, parse_dates=True)

    X_train = train_df[feature_cols].to_numpy()
    X_val = val_df[feature_cols].to_numpy()

    model_matrix = np.empty((0, len(feature_cols) + 1))
    y_actual = [None for _ in range(len(forecast_horizons))]
    T_baseline = [None for _ in range(len(forecast_horizons))]

    for i in forecast_horizons:
        target_col = "T_plus" + str(i)

        y_train = train_df[target_col].to_numpy()
        y_val = val_df[target_col].to_numpy()
        T_baseline[forecast_horizons.index(i)] = val_df["T"].to_numpy()

        beta = fit_linear_regression(X_train, y_train)
        y_actual[forecast_horizons.index(i)] = y_val
        model_matrix = np.vstack([model_matrix, beta])

    y_predicted = [np.array([]) for _ in range(len(forecast_horizons))]

    for i in X_val:
        for g in range(len(forecast_horizons)):
            y_predicted[g] = np.append(y_predicted[g], predict(model_matrix, i)[g])

    for j in range(len(forecast_horizons)):
        model_root_error = rmse(y_actual[j], y_predicted[j])
        model_absolute_error = mae(y_actual[j], y_predicted[j])
        baseline_root_error = rmse(y_actual[j], T_baseline[j])
        baseline_absolute_error = mae(y_actual[j], T_baseline[j])

        print(str(forecast_horizons[j]) + "-hour temperature forecast errors:")
        print("Model Root Mean Square Error (RMSE):", model_root_error)
        print("Model Mean Absolute Error (MAE):", model_absolute_error)
        print("Baseline Root Mean Square Error (RMSE):", baseline_root_error)
        print("Baseline Mean Absolute Error (MAE):", baseline_absolute_error, '\n')

    for i in range(len(forecast_horizons)):
        plot_df = pd.DataFrame({
            "Actual": y_actual[i],
            "Model": y_predicted[i],
        }, index=val_df.index)

        plt.figure(figsize=(10, 4))
        plot_df.plot(ax=plt.gca(), linewidth=1)
        plt.title(str(forecast_horizons[i]) + "-hour temperature forecast: validation sample")
        plt.ylabel("T [°C]")
        plt.tight_layout()
        plt.show()