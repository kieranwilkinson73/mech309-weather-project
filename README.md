# MECH 309 – Weather Prediction Project: PART I

## Overview
This project develops a data-driven model to predict near-surface air temperature in Montréal using historical weather data.
It is based on linear regression and evaluates predictions at multiple forecast horizons (1, 3, 6, 12, and 24 hours).

## Project Structure:
The repository is organised in the following way:

-data/
    -raw/
     contains raw weather data if used.
    -processed/
    containts processed data used by the model, including 'train_weather.csv' (the data used to train the model) and 'val_weather.csv' (the data used to validate the model's performance)
-src/
    -final_temperature_model.py
    Loads the processed datasets, trains the regression model, evaluates forecasting predictions, draws plots
    -other files used for intermediary data procssing/model development.
-figures/
    contains plots generated for the report
-results/
    contains results from our model as CSV files.

## Data Source
Weather data is retrieved using the Open-Meteo API:
https://open-meteo.com/

Data includes:
- Temperature (2m)
- Wind speed (10m)

## How to Run
