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
    forecast_horizons = [1, 3, 6, 12, 24] # Various forecast horizons to be modeled

    # Define feature columns as columns whose data is used with respective preconditioner variables
    feature_cols = [
        'T', 'W', 'sin_day', 'cos_day', 'sin_year', 'cos_year',
        'RH', 'P', 'Prec', 'Cloud', 'T_lag1', 'T_lag3', 'T_lag12'
    ]

    # Retrieve dataframs for testing and validation
    train_df = pd.read_csv("data/processed/train_weather.csv", index_col=0, parse_dates=True) # Define training dataframe from training CSV file
    val_df = pd.read_csv("data/processed/val_weather.csv", index_col=0, parse_dates=True) # Define validating dataframe from validating CSV file

    # Convert Dataframe data to useable numpy data
    X_train = train_df[feature_cols].to_numpy() # Assign the training matrix data from preprocessed dataframe file to useable numpy matrix
    X_val = val_df[feature_cols].to_numpy() # Assign the validating matrix data from preprocessed dataframe file to useable numpy matrix

    # Creating empy matrices/arrays to store future result and validating data
    model_matrix = np.empty((0, len(feature_cols) + 1)) # Define an empty matrix to store final model
    y_actual = [None for _ in range(len(forecast_horizons))] # Create list to store actual y values during validation period
    T_baseline = [None for _ in range(len(forecast_horizons))] # Create list to store baseline temperature values for validation period

    # Loop through different forecast horizons and create model line by line
    for i in forecast_horizons:
        target_col = "T_plus" + str(i) # Set the target column as the current forecast horizon

        # Create target temperatures for training and validation
        y_train = train_df[target_col].to_numpy() # Assign the training temperature values from dataframe file to a useable numpy array
        y_val = val_df[target_col].to_numpy() # Assign the validating temperature values from dataframe file to a useable numpy array
        T_baseline[forecast_horizons.index(i)] = val_df["T"].to_numpy() # Create a baseline temperature dataset to compare the models predictions to

        # Model calculation
        beta = fit_linear_regression(X_train, y_train) # Calculate the beta variables associated with each predictor variable (weight) via least square method
        y_actual[forecast_horizons.index(i)] = y_val # Add the current set of correct y values to later use for validation
        
        # Append row to model matrix
        model_matrix = np.vstack([model_matrix, beta])

    # Save the model matrix to a csv file
    np.savetxt("src/matrix_model.csv", model_matrix, delimiter=",", fmt="%.18e")

    # Create list of lists to store the various predicted temperatures for each forecast horizon
    y_predicted = [np.array([]) for _ in range(len(forecast_horizons))]

    # Calculate the predicted temperature values
    for i in X_val: # Looping through the dataframe of predictor variables
        for g in range(len(forecast_horizons)): # Looping through each forecast horizon for each set of datapoints
            y_predicted[g] = np.append(y_predicted[g], predict(model_matrix, i)[g]) # Predict temperature for set horizon and append it to list of results

    # Error Calculation and Display
    row_headers = [str(i)+"h" for i in forecast_horizons] # Create Row Headers for output CSV file
    column_headers = ["Model RMSE [\N{DEGREE SIGN}C]", "Model MAE [\N{DEGREE SIGN}C]", "Baseline RMSE [\N{DEGREE SIGN}C]", "Baseline MAE [\N{DEGREE SIGN}C]"] # Create Column headers for output CSV file
    error_results = np.empty((len(forecast_horizons), 4)) # Create an empy matrix to store error values
    for j in range(len(forecast_horizons)): # Loop through forecast horizons to calculate errors for each
        error_results[j, 0] = rmse(y_actual[j], y_predicted[j]) # Calculate the root mean square error of the predicted temperatures to the validation temperatures
        error_results[j, 1] = mae(y_actual[j], y_predicted[j]) # Calculate the mean absolute error of the predicted temperatures to the validation temperatures
        error_results[j, 2] = rmse(y_actual[j], T_baseline[j]) # Calculate the root mean square error of the baseline temperatures to the validation temperatures
        error_results[j, 3] = mae(y_actual[j], T_baseline[j]) # Calculate the mean absolute error of the baseline temperatures to the validation temperatures        
    error_df = pd.DataFrame(error_results, index=row_headers, columns=column_headers) # Convert results numpy matrix to pandas dataframe for easier exporting
    error_df.to_csv("src/error_results.csv", float_format="%.18e", index_label="Forecast Horizon") # Create csv file with error results

    # Plotting the actual temperature against the predicted temperature
    for i in range(len(forecast_horizons)):
        plot_df = pd.DataFrame({
            "Actual": y_actual[i],
            "Model": y_predicted[i],
            # "Baseline": T_baseline[i]
        }, index=val_df.index)

        plt.figure(figsize=(10, 4))
        plot_df.plot(ax=plt.gca(), linewidth=1)
        plt.title(str(forecast_horizons[i]) + "-hour temperature forecast: validation sample")
        plt.ylabel("T [°C]")
        plt.tight_layout()
        plt.show()
