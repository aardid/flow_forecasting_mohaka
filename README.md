# Flow Forecasting in the Mohaka Catchment

## Overview
This repository contains a Python-based machine learning pipeline for forecasting river flow in the Mohaka catchment, New Zealand. The model uses upstream flow and rainfall data to predict downstream flow at the McVicars station. This work is part of a study on flood forecasting using data-driven approaches.

## Installation Guide

### 1. Clone the Repository
To get started, clone this repository to your local machine:
```bash
git clone https://github.com/aardid/flow_forecasting_mohaka.git
cd flow_forecasting_mohaka
```

### 2. Set Up the Environment
Ensure you have Conda installed. Then, create and activate the environment using the provided YAML file:
```bash
conda env create -f lab_env_flood.yml
conda activate lab_env_flood
```

## Dataset
The dataset includes flow and rainfall measurements from three stations:
- **Taharua** (rainfall and flow data)
- **Porponui** (flow data)
- **McVicars** (downstream flow data, target for forecasting)

Files included:
- `observedflow_Taharua.csv`
- `observedflow_Porponui.csv`
- `observedflow_McVicars.csv`
- `Taharua_PCP.xlsx`

## Running the Forecasting Script
The main script `main.py` runs the forecasting pipeline. Execute it with:
```bash
python main.py
```
The script follows a structured pipeline:

### Step 1: Import and Plot Data
- Load rainfall and flow data for all stations.
- Visualize the data for exploration.

### Step 2: Generate Time-Delayed Features
- Create lagged variables to capture dependencies over time.

### Step 3: Define Features and Target
- Specify input features and the target variable for forecasting.
- Normalize features for improved model performance.

### Step 4: Split Data into Training and Testing Sets
- Reserve a portion of the data for evaluation.

### Step 5: Train and Evaluate a Neural Network Model
- Use a Multi-Layer Perceptron (MLP) to model flow dynamics.
- Assess performance using Mean Squared Error (MSE) and Mean Absolute Error (MAE).

### Step 6: Compare Multiple Models
- Train and evaluate different models to select the best performer.

### Step 7: Forecast Flow at McVicars
- Use the best model to predict flow at the McVicars station.

### Step 8: Explore the Impact of Data Availability
- Train models with different data constraints (e.g., removing telemetry data, reducing station coverage) to assess how accuracy is affected.

### Step 9: Hyperparameter Tuning
- Optimize model parameters to improve predictive accuracy.

## Results and Visualization
The script generates plots comparing predicted vs. actual flows and assessing the impact of different data availability scenarios.

## Contributions
Author: **Dr. Alberto Ardid**  
Email: alberto.ardid@canterbury.ac.nz

## License
This project is licensed under the MIT License.

