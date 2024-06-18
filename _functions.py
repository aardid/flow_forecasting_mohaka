import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Import and plot flow data for a station
def import_and_plot_flow_data(flow_data_file, station, plot=False):
    """Import and plot flow data for a station."""
    flow_data = pd.read_csv(flow_data_file)
    flow_data['Date'] = pd.to_datetime(flow_data['Date'], format='%d/%m/%Y')
    flow_data.set_index('Date', inplace = True)
    flow_data.sort_index(inplace=True)  # Sort by date
    flow_data.clip(lower=0, inplace = True)  # Remove negative values
    flow_data.interpolate(method='linear', inplace=True)  # Interpolate missing values
    flow_data.rename(columns={'Flow': f'Flow_{station}'}, inplace=True)  # Rename column
    if plot:
        # Plot flow data
        plt.figure(figsize=(10, 2))
        plt.plot(flow_data.index, flow_data[f'Flow_{station}'], label=f'Flow_{station}')
        plt.title(f'Flow Data - {station}')
        plt.xlabel('Date')
        plt.ylabel('Flow (m3/s)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return flow_data

# Merge data
def merge_data(data_list):
    """Merge multiple DataFrames."""
    merged_data = pd.concat(data_list, axis=1)
    merged_data = merged_data.dropna()
    return merged_data

# Define features and target variables
def define_features_target(data):
    """Define features and target variables."""
    X = data.filter(like='_t-')  # Select columns with '_t-' in their names (indicating delayed data)
    y  = data['Flow_McVicars']  # Target variable: Flow from McVicars
    return X, y

# normalize features
def normalize_features(X):
    """
    Normalize features using StandardScaler.

    Parameters:
    - X: Input features.

    Returns:
    - X_scaled: Normalized features.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Import, clean, and plot rain data for a station
def import_and_plot_rain_data(rain_data_file, station, plot=False):
    """Import, clean, and plot rain data for a station."""
    rain_data = pd.read_excel(rain_data_file)#, parse_dates=['Date'], index_col='Date')
    rain_data['Date'] = pd.to_datetime(rain_data['date'], format='%d/%m/%Y')
    rain_data.set_index('Date', inplace = True)
    rain_data = rain_data.drop('date', axis=1)
    rain_data.sort_index(inplace=True)  # Sort by date
    rain_data.clip(lower=0, inplace = True)  # Remove negative values
    rain_data.interpolate(method='linear', inplace=True)  # Interpolate missing values
    rain_data.rename(columns={'rain': f'Rain_{station}'}, inplace=True)  # Rename column

    if plot:
        # Plot rain data
        plt.figure(figsize=(10, 3))
        plt.plot(rain_data.index, rain_data[f'Rain_{station}'], color='c',label=f'Rain_{station}')
        plt.title(f'Rain Data - {station}')
        plt.xlabel('Date')
        plt.ylabel('Rain (mm)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return rain_data

# Calculate time delayed data
def calculate_time_delayed_data(data, delay, plot=False, plot_period=20):
    """Calculate time delayed data."""
    delayed_data = data.copy()
    for col in delayed_data.columns:
        if col != 'Date':  # Skip 'Date' column
            for i in range(1, delay + 1):
                delayed_data[f'{col}_t-{i}'] = data[col].shift(i)
    
    delayed_data.dropna(inplace=True)
    
    if plot:
        fig, ax = plt.subplots(figsize=(12, 2))
        plot_data = delayed_data.iloc[-plot_period:]
        for col in plot_data.columns:
            if '_t-' in col:
                ax.plot(plot_data.index, plot_data[col], label=col)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.set_title('Time Delayed Time Series')
        ax.legend()
        plt.grid(True)
        plt.show()
    
    return delayed_data

# Define features and target variables
def define_features_target_current(data):
    """Define features and target variables."""
    # Select columns with '_t-' in their names (indicating delayed data)
    X_delayed = data.filter(like='_t-')

    # Include current non-delayed data from Taharua (rain and flow) and Porponui (flow)
    X_current = pd.concat([data['Rain_Taharua'],  # Select the last column for current rain data
                           data['Flow_Taharua'],  # Select the last column for current flow data from Taharua
                           data['Flow_Porponui']],  # Select the last column for current flow data from Porponui
                          axis=1)

    # Combine delayed and current data
    X = pd.concat([X_current, X_delayed], axis=1)

    # Target variable: Flow from McVicars
    y = data['Flow_McVicars']

    return X, y

# Split data into training and testing sets
def split_data(X, y, test_size=0.3):
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    return X_train, X_test, y_train, y_test

# Train a neural network
def train_neural_network(X_train, y_train):
    """Train a neural network."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=1000, random_state=41)
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Train a random forest
def train_random_forest(X_train, y_train):
    """Train a random forest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=41)
    model.fit(X_train, y_train)
    return model

# Train a linear regression model
def train_linear_regression(X_train, y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Predict and evaluate model and show scatter plots
def predict_evaluate_model(model, scaler, X_test, y_test, plot = False):
    """Predict and evaluate model, and show scatter plots."""
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    if plot:
        # Plot scatter plot of observed vs predicted flow
        plt.figure(figsize=(4, 4))
        plt.scatter(y_test, y_pred)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red')
        plt.xlabel('Observed Flow (m3/s)')
        plt.ylabel('Predicted Flow (m3/s)')
        plt.title('Observed vs Predicted Flow')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    
    return mse, mae

# Explore different models and select the best model
def explore_models_select_best(X_train, y_train, X_test, y_test):
    """Explore different models and select the best model."""
    # Here, you can explore different models and select the best one based on evaluation metrics
    # For simplicity, let's just compare the models based on Mean Absolute Error (MAE)

    # Neural Network
    model_nn, scaler_nn = train_neural_network(X_train, y_train)
    mse_nn, mae_nn = predict_evaluate_model(model_nn, scaler_nn, X_test, y_test, plot=False)

    # Train a neural network with log-transformed target variable
    #model_nn_log, scaler_nn_log = train_neural_network(X_train, np.log1p(y_train))
    #mse_nn_log, mae_nn_log = predict_evaluate_model(model_nn_log, scaler_nn_log, X_test, y_test)

    # Random Forest
    model_rf = train_random_forest(X_train, y_train)
    mse_rf, mae_rf = predict_evaluate_model(model_rf, None, X_test, y_test, plot=False)

    # Linear Regression
    model_lr = train_linear_regression(X_train, y_train)
    mse_lr, mae_lr = predict_evaluate_model(model_lr, None, X_test, y_test, plot=False)

    # Create a dictionary to store the results
    results = {
        'Neural Network': {'MAE': mae_nn},
        'Random Forest': {'MAE': mae_rf},
        'Linear Regression': {'MAE': mae_lr},
        #'Neural Network (Log)': {'MAE': mae_nn_log}
    }

    # Print the results
    print("\nModel Evaluation Results:")
    print("==========================")
    for model, metrics in results.items():
        print(f"{model}: MAE = {metrics['MAE']:.2f}")

    # Select the best model based on MAE
    best_model = min(results, key=lambda x: results[x]['MAE'])
    print(f"\nBest Model: {best_model}")

    # Return the best model and its metrics
    return best_model, results[best_model]['MAE']

# Explore hyperparameters of the best model to reduce MAE
def explore_hyperparameters(X_train, y_train, X_test, y_test, model_name):
    """Explore hyperparameters of the specified model to reduce MAE."""
    if model_name == 'Neural Network':
        # Define hyperparameters grid for neural network
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam']
        }
        model = MLPRegressor(max_iter=1000, random_state=1)
    elif model_name == 'Random Forest':
        # Define hyperparameters grid for random forest
        param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor(random_state=1)
    elif model_name == 'Linear Regression':
        # Linear Regression doesn't have hyperparameters to tune
        print("Linear Regression doesn't have hyperparameters to tune.")
        return model_name, None, None
    elif model_name == 'Neural Network (Log)':
        # Define hyperparameters grid for neural network with log-transformed target variable
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam']
        }
        model = MLPRegressor(max_iter=1000, random_state=1)

    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3, n_jobs=8)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Include model name in best_params
    #best_params['model'] = model_name

    # Retrain the model with the best hyperparameters
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = best_model.predict(X_test)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)

    # Print best hyperparameters and MAE
    print(f"\nBest Hyperparameters for {model_name}\n====================\n{best_params}\n")
    print(f"MAE with Best Hyperparameters for {model_name}: {mae:.2f}")

    return model_name, best_model, best_params

# Train a model with the best model and predict the flow out of sample (in mcvicars)
def train_predict_best_model(merged_data, target_station, best_model, plot_training=False):
    """Train a model with the best model and predict the flow out of sample (in McVicars)."""

    ## Task 4: Define features and target variables
    X, y = define_features_target(merged_data)

    ## Task 5: Split data into training and testing sets
    def _split_data(X, y, test_size=0.3):
        """Split data into training and testing sets."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = _split_data(X, y)

    # Initialize X_train_scaled and scaler
    X_train_scaled = None
    scaler = None

    # Train the best model
    if best_model == 'Neural Network':
        model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=1000, random_state=1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
    elif best_model == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=1)
        model.fit(X_train, y_train)
    elif best_model == 'Linear Regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif best_model == 'Neural Network (Log)':
        model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', max_iter=1000, random_state=1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, np.log1p(y_train))

    # Predict the flow in McVicars
    X_test_scaled = X_test  # By default, no scaling needed if not a neural network model
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)  # Scale test data if scaler is defined

    if best_model == 'Neural Network (Log)':
        y_pred_mcvicars = np.expm1(model.predict(X_test_scaled))
    else:
        y_pred_mcvicars = model.predict(X_test_scaled)

    # Predict the flow in McVicars for the training set
    if best_model == 'Neural Network' or best_model == 'Neural Network (Log)':
        X_train_scaled = scaler.transform(X_train)
        y_pred_train = np.expm1(model.predict(X_train_scaled))
    else:
        y_pred_train = model.predict(X_train)

    # Calculate MSE and MAE for the test set
    mse = mean_squared_error(y_test, y_pred_mcvicars)
    mae = mean_absolute_error(y_test, y_pred_mcvicars)

    # Calculate MSE and MAE for the training set
    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)

    # Print MSE and MAE for the test set
    print(f"\nTest MSE for {best_model}: {mse:.2f}")
    print(f"Test MAE for {best_model}: {mae:.2f}")

    # Print MSE and MAE for the training set
    print(f"\nTrain MSE for {best_model}: {mse_train:.2f}")
    print(f"Train MAE for {best_model}: {mae_train:.2f}")

    # Plot observed vs predicted flow for both train and test sets
    plt.figure(figsize=(10, 3))
    plt.plot(y_test.index, y_test, label='Observed Flow (Test)', color='gray')
    plt.plot(y_test.index, y_pred_mcvicars, label='Predicted Flow (Test)', linestyle='-', color='orange')
    if plot_training:
        plt.plot(y_train.index, y_train, label='Observed Flow (Train)', color='gray')
        plt.plot(y_train.index, y_pred_train, label='Predicted Flow (Train)', linestyle='--', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Flow (m3/s)')
    plt.title(f'Observed vs Predicted Flow in {target_station}\n Model {best_model}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def include_current_data(merged_data, rain_data_taharua, flow_data_taharua, flow_data_porponui):
    """Include current non-delayed data from Taharua (rain and flow) and Porponui (flow) in the merged data."""
    # Combine the current non-delayed data with the merged data
    merged_data_with_current = merged_data.copy()
    merged_data_with_current = pd.concat([merged_data_with_current, rain_data_taharua, flow_data_taharua, flow_data_porponui], axis=1)

    # Drop rows with NaN values (if any)
    merged_data_with_current.dropna(inplace=True)

    return merged_data_with_current

def plot_forecasts(merged_data, target_station, best_model_nn, best_model_rf, best_model_lr):
    """Plot forecasts from the three best models."""
    
    # Define features and target variables
    X, y = define_features_target(merged_data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Initialize X_train_scaled
    X_train_scaled = None

    # Train and predict with Neural Network if it's the best model
    model_name, best_model_nn, best_params_nn = best_model_nn
    if model_name == 'Neural Network':
        model_nn = MLPRegressor(**best_params_nn, max_iter=1000, random_state=1)
        scaler_nn = StandardScaler()
        X_train_scaled = scaler_nn.fit_transform(X_train)
        model_nn.fit(X_train_scaled, y_train)
        X_test_scaled_nn = scaler_nn.transform(X_test)
        y_pred_nn = model_nn.predict(X_test_scaled_nn)
    elif model_name == 'Neural Network (Log)':
        model_nn = MLPRegressor(**best_params_nn, max_iter=1000, random_state=1)
        scaler_nn = StandardScaler()
        X_train_scaled = scaler_nn.fit_transform(X_train)
        model_nn.fit(X_train_scaled, np.log1p(y_train))
        X_test_scaled_nn = scaler_nn.transform(X_test)
        y_pred_nn = np.expm1(model_nn.predict(X_test_scaled_nn))
    else:
        print("Neural Network is not the best model selected.")
        y_pred_nn = None

    # Train and predict with Random Forest if it's the best model
    model_name, best_model_rf, best_params_rf = best_model_rf
    if model_name == 'Random Forest':
        model_rf = RandomForestRegressor(**best_params_rf, random_state=1)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
    else:
        print("Random Forest is not the best model selected.")
        y_pred_rf = None

    # Train and predict with Linear Regression if it's the best model
    model_name, best_model_lr, best_params_lr = best_model_lr
    if model_name == 'Linear Regression':
        model_lr = LinearRegression()
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)
    else:
        print("Linear Regression is not the best model selected.")
        y_pred_lr = None

    # Calculate Mean Absolute Error (MAE) for each model
    mae_nn = mean_absolute_error(y_test, y_pred_nn) if y_pred_nn is not None else None
    mae_rf = mean_absolute_error(y_test, y_pred_rf) if y_pred_rf is not None else None
    mae_lr = mean_absolute_error(y_test, y_pred_lr) if y_pred_lr is not None else None

    # Plot forecasts
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 3))

    # Plot top subplot with observed data and forecasts
    ax1.plot(y_test.index, y_test, label='Observed', color='gray', marker='.', alpha=.7)
    if y_pred_nn is not None:
        ax1.plot(y_test.index, y_pred_nn, label=f'Neural Network Forecast (MAE: {mae_nn:.2f})', linestyle='-', color='blue', alpha=.7)
    if y_pred_rf is not None:
        ax1.plot(y_test.index, y_pred_rf, label=f'Random Forest Forecast (MAE: {mae_rf:.2f})', linestyle='-', color='green', alpha=.7)
    if y_pred_lr is not None:
        ax1.plot(y_test.index, y_pred_lr, label=f'Linear Regression Forecast (MAE: {mae_lr:.2f})', linestyle='-', color='red', alpha=.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Flow (m3/s)')
    ax1.set_title(f'Observed vs Forecasted Flow in {target_station}')
    ax1.legend()
    ax1.grid(True)

    plt.tight_layout()
    # Save the figure as a PNG file
    plt.savefig('forecast_plot_models.png')
    plt.show()

    # Plot forecasts
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))

    # Plot bottom subplot with difference between forecasts and observed data
    ax2 = axs[0]
    if y_pred_nn is not None:
        ax2.plot(y_test.index, y_pred_nn - y_test, label='Neural Network Difference', linestyle='-', color='blue', alpha=.9)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Difference (m3/s)')
    ax2.set_title(f'Difference Between Forecasts and Observed Flow in {target_station}')
    ax2.set_ylim([-np.max(y_test), np.max(y_test)])
    ax2.legend()
    ax2.grid(True)

    ax3 = axs[1]
    if y_pred_rf is not None:
        ax3.plot(y_test.index, y_pred_rf - y_test, label='Random Forest Difference', linestyle='-', color='green', alpha=.9)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Difference (m3/s)')
    ax3.set_title(f'Difference Between Forecasts and Observed Flow in {target_station}')
    ax3.set_ylim([-np.max(y_test), np.max(y_test)])
    ax3.legend()
    ax3.grid(True)

    ax4 = axs[2]
    if y_pred_lr is not None:
        ax4.plot(y_test.index, y_pred_lr - y_test, label='Linear Regression Difference', linestyle='-', color='red', alpha=.9)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Difference (m3/s)')
    ax4.set_title(f'Difference Between Forecasts and Observed Flow in {target_station}')
    ax4.set_ylim([-np.max(y_test), np.max(y_test)])
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    # Save the figure as a PNG file
    plt.savefig('forecast_plot_diferenes.png')
    plt.show()