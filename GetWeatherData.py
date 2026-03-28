from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float
    timezone: str


MONTREAL = Location(
    name="Montreal, QC",
    lat=45.5017,
    lon=-73.5673,
    timezone="America/Montreal",
)

def fetch_open_meteo_hourly(
    start_date: str,
    end_date: str,
    location: Location = MONTREAL,
    hourly_vars: List[str] | None = None,
) -> pd.DataFrame:
    if hourly_vars is None:
        hourly_vars = [ # Removed some for basic model, will re-add in future iterations
            "temperature_2m",
            "wind_speed_10m",
            # "wind_direction_10m",
            # "relative_humidity_2m",
            # "surface_pressure",
            # "precipitation",
            # "cloud_cover",
        ]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": location.timezone,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    hourly = payload.get("hourly", {})
    times = hourly.get("time", None)
    if times is None:
        raise RuntimeError(f"Open-Meteo response missing 'hourly.time'. Keys: {payload.keys()}")

    idx = pd.to_datetime(times)
    df = pd.DataFrame(index=idx)
    for k, v in hourly.items():
        if k == "time":
            continue
        df[k] = v
    df.index.name = "time_local"
    return df

# Preprocessing + features
def preprocess(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame: # Added horizons variable to add all target
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=df.index.tz)
    df = df.reindex(full_idx)

    df = df.interpolate(limit=6)
    df = df.ffill().bfill()


    # Removed some for basic model, will re-add in future iterations
    rename = {
        "temperature_2m": "T",
        "wind_speed_10m": "W",
        # "wind_direction_10m": "Wd",
        # "relative_humidity_2m": "RH",
        # "surface_pressure": "P",
        # "precipitation": "Prec",
        # "cloud_cover": "Cloud",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Diurnal + seasonal features
    hour = df.index.hour.to_numpy()
    omega = 2 * math.pi / 24.0
    # TODO: Student can add whatever you like to the dataset here. Example here
    df["sin_day"] = np.sin(omega * hour)
    df["cos_day"] = np.cos(omega * hour)

    doy = df.index.dayofyear.to_numpy()
    omega_y = 2 * math.pi / 365.25
    # TODO: Student can add whatever you like to the dataset here

    # Model periodic day of year
    df["sin_year"] = np.sin(omega_y * doy)
    df["cos_year"] = np.cos(omega_y * doy)

    # Add lag for 1 hour forecast
    add_lags(df, 'T', [1])

    for i in horizons:
        # Create the various forecast horizon columns in the data
        # df2 = df.copy()
        # df2["T_plus"+str(i)] = df2["T"].shift(-i)
        # df = pd.concat([df, df2], ignore_index=True)

        df["T_plus"+str(i)] = df["T"].shift(-i)

    # Define feature columns as columns whose data is used with respective preconditioner variables
    feature_columns = ['T', 'W', 'sin_day', 'cos_day', 'sin_year', 'cos_year', 'T_lag1']

    # Remove any empty rows
    df = df.dropna(subset=feature_columns + ['T_plus' + str(c) for c in horizons]).copy()

    # Return the dataframe and feature columns
    return df, feature_columns

# TODO: Students you might find this function useful
# Will use Lags in future iterations, currently want to test a simpler model without historical data
def add_lags(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    for L in lags:
        if L <= 0:
            continue
        df[f"{col}_lag{L}"] = df[col].shift(L)
    return df

# TODO: Students you might find this function useful
def split_train_val(data: pd.DataFrame, val_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(data) <= val_hours + 10:
        raise ValueError("Not enough samples for requested validation window.")
    return data.iloc[:-val_hours].copy(), data.iloc[-val_hours:].copy()

# Function to find least squares solution
def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_aug = np.column_stack([np.ones(len(X)), X]) # Create an extra column of 1s to create the intercept term
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None) # Solve for the beta variables using least squares method in numpy linalg library
    return beta # Return the beta variables

# Use regression model to predict y
def predict_linear_regression(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    X_aug = np.column_stack([np.ones(len(X)), X]) # Define the matrix with an extra column of 1s to include the intercept term
    return X_aug @ beta # Calculate y_hat given the matrix and beta variables

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
    start_date = "2024-06-01"
    end_date = "2025-08-30"
    val_hours = 24*14 # Using last 14 days for validation
    forecast_horizons = [1, 3, 6, 12, 24] # Various forecast horizons to be modeled
    
    montreal = Location(
        name="Montreal, QC",
        lat=45.5017,
        lon=-73.5673,
        timezone="America/Montreal",
    )

    # Fetching and pre-processing data
    print(f"Fetching Open-Meteo hourly data for {montreal.name}...")
    df_raw = fetch_open_meteo_hourly(start_date, end_date, location=montreal)
    print("Preprocessing...")
    df, feature_cols = preprocess(df_raw, forecast_horizons)

    # Plotting the pre-processed temperature data
    plt.figure()
    df["T"].plot(linewidth=1)
    plt.title("Montreal hourly temperature (2m)")
    plt.ylabel("T [°C]")
    plt.tight_layout()
    plt.show()

    # Define an empty matrix
    model_matrix = np.empty((0, len(feature_cols)+ 1)) # Set to 8 for now, maybe add a way that auto-calculates

    train_df, val_df = split_train_val(df, val_hours=val_hours) # Split data region into training and validation
    X_val = val_df[feature_cols].to_numpy() # Assign the validating matrix data from preprocessed dataframe file to useable numpy matrix
    y_actual = [None for _ in range(len(forecast_horizons))] # Create list to store actual y values during validation period


    # Loop through different forecast horizons and create one model for each
    for i in forecast_horizons:
        target_col = "T_plus" + str(i) # Set the target column as the current forecast horizon
        
        # Organizing data
        train_df, val_df = split_train_val(df, val_hours=val_hours) # Split data region into training and validation
        X_train = train_df[feature_cols].to_numpy() # Assign the training matrix data from preprocessed dataframe file to useable numpy matrix
        y_train = train_df[target_col].to_numpy() # Assign the training temperature values from dataframe file to a useable numpy array
        X_val = val_df[feature_cols].to_numpy() # Assign the validating matrix data from preprocessed dataframe file to useable numpy matrix
        y_val = val_df[target_col].to_numpy() # Assign the validating temperature values from dataframe file to a useable numpy array

        # Model calculation
        beta = fit_linear_regression(X_train, y_train) # Calculate the beta variables associated with each predictor variable (weight) via least square method
        yhat_val = predict_linear_regression(X_val, beta) # Calculate the predicted temperature values within the validation period
        y_actual[forecast_horizons.index(i)] = y_val # Add the current set of correct y values to later use for validation
        
        # Append row to model matrix
        model_matrix = np.vstack([model_matrix, beta])

        # Error calculation
        error_rms = rmse(y_val, yhat_val) # Calculate the root mean square error of the predicted temperatures to the validation temperatures
        error_ma = mae(y_val, yhat_val) # Calculate the mean absolute error of the predicted temperatures to the validation temperatures
        print(str(i)+"-hour temperature forecast errors:")
        print('Root Mean Square Error (RMSE):', error_rms) # Display RMSE in the terminal
        print('Mean Absolute Error (MAE):', error_ma, '\n') # Display MAE in the terminal

        # Plotting the actual temperature against the predicted temperature
        # plot_df = pd.DataFrame({
        #             "Actual": y_val,
        #             "Model": yhat_val,
        #         }, index=val_df.index)
        # plt.figure(figsize=(10, 4))
        # plot_df.iloc[:7 * 24].plot(ax=plt.gca(), linewidth=1)
        # plt.title(str(i)+"-hour temperature forecast: validation sample")
        # plt.ylabel("T [°C]")
        # plt.tight_layout()
        # # plt.show()
    # print(model_matrix)

    # Create list of lists to store the various predicted temperatures for each forecast horizon
    y_predicted = [np.array([]) for _ in range(len(forecast_horizons))]

    # Calculate the predicted temperature values
    for i in X_val: # Looping through the dataframe of predictor variables
        for g in range(len(forecast_horizons)): # Looping through each forecast horizon for each set of datapoints
            y_predicted[g] = np.append(y_predicted[g], predict(model_matrix, i)[g]) # Predict temperature for set horizon and append it to list of results


    # Plotting the actual temperature against the predicted temperature
    for i in range(len(forecast_horizons)):
        plot_df = pd.DataFrame({
                    "Actual": y_actual[i],
                    "Model": y_predicted[i],
                }, index=val_df.index)
        plt.figure(figsize=(10, 4))
        plot_df.iloc[:7 * 24].plot(ax=plt.gca(), linewidth=1)
        plt.title(str(forecast_horizons[i])+"-hour temperature forecast: validation sample")
        plt.ylabel("T [°C]")
        plt.tight_layout()
        plt.show()
