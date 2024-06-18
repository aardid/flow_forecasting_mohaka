__author__ = """Alberto Ardid"""
__email__ = 'alberto.ardid@canterbury.ac.nz'

# Script for paper 'Modeling River Flow Data Using Machine Learning for Flood Forecasting'

# PURPOSE:
# Use upstream flow and rain date to train machine learning models for forecast downstream.

# INSTRUCTION:
# - use code-folding to collapse all code in this file (VS Code: Ctrl+k, Ctrl+0)
# - unfold the 'main' function - work through the TASKs one at a time

from _functions import *

def main():
    # Flow and rain data file paths
    flow_data_file_taharua = 'observedflow_Taharua.csv'
    flow_data_file_porponui = 'observedflow_Porponui.csv'
    flow_data_file_mcvicars = 'observedflow_McVicars.csv'
    rain_data_file_taharua = 'Taharua_PCP.xlsx'

    ## Task 1: Import and plot rain and flow data for the three stations
    rain_data_taharua = import_and_plot_rain_data(rain_data_file_taharua, station='Taharua', plot=True)
    flow_data_taharua = import_and_plot_flow_data(flow_data_file_taharua, station='Taharua', plot=True)
    flow_data_porponui = import_and_plot_flow_data(flow_data_file_porponui, station='Porponui', plot=True)
    flow_data_mcvicars = import_and_plot_flow_data(flow_data_file_mcvicars, station='McVicars', plot=True)
    
    ## Task 2: Calculate time delayed data
    delayed_rain_data_taharua = calculate_time_delayed_data(rain_data_taharua, delay=5)
    delayed_flow_data_taharua = calculate_time_delayed_data(flow_data_taharua, delay=5, plot=False, plot_period=20)
    delayed_flow_data_porponui = calculate_time_delayed_data(flow_data_porponui, delay=5)
    delayed_flow_data_mcvicars = calculate_time_delayed_data(flow_data_mcvicars, delay=5)
    merged_data = merge_data([delayed_rain_data_taharua, delayed_flow_data_taharua, delayed_flow_data_porponui, delayed_flow_data_mcvicars])
    
    ## Task 3: Define features and target variables
    X, y = define_features_target(merged_data)
    ## Normalize features
    X_scaled = normalize_features(X)

    ## Task 4: Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    ## Task 5: Train and Evaluate a Neural Network Model
    ## Neural Network
    model_nn, scaler_nn = train_neural_network(X_train, y_train)
    mse_nn, mae_nn = predict_evaluate_model(model_nn, scaler_nn, X_test, y_test, plot=True)
    print("Neural Network:")
    print("MSE:", mse_nn)
    print("MAE:", mae_nn)

    ## Task 6: Train and Evaluate a Several Models
    best_model, mae_best = explore_models_select_best(X_train, y_train, X_test, y_test)

    ## Task 7: Train a model with the best model and predict the flow out of sample (in mcvicars)
    train_predict_best_model(merged_data, target_station='McVicars', best_model=best_model, plot_training=True)

    ## Task 8: Include current data from Taharua and Porponui 
    X, y = define_features_target_current(merged_data)
    X_scaled = normalize_features(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    best_model, mae_best = explore_models_select_best(X_train, y_train, X_test, y_test)
    train_predict_best_model(merged_data, target_station='McVicars', best_model=best_model)

    ## Task 9: Explore hyperparameters for all models and evaluate  
    model_name_nn, best_model_nn, best_params_nn = explore_hyperparameters(X_train, y_train, X_test, y_test, 'Neural Network')
    model_name_rf, best_model_rf, best_params_rf = explore_hyperparameters(X_train, y_train, X_test, y_test, 'Random Forest')
    model_name_lr, best_model_lr, best_params_lr = explore_hyperparameters(X_train, y_train, X_test, y_test, 'Linear Regression')
    ## Plot forecasts from the best models
    plot_forecasts(merged_data, 'McVicars', (model_name_nn, best_model_nn, best_params_nn),
                    (model_name_rf, best_model_rf, best_params_rf), (model_name_lr, best_model_lr, best_params_lr))

# Main function
if __name__ == '__main__':
    main()