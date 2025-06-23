import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle
import os

def create_derived_features(df):
    """Create additional meaningful features from existing ones."""
    if all(col in df.columns for col in ['cylinder_head_temp', 'exhaust_gas_temp', 'bearing_temp']):
        df['temp_diff_head_exhaust'] = df['exhaust_gas_temp'] - df['cylinder_head_temp']
        df['temp_diff_head_bearing'] = df['cylinder_head_temp'] - df['bearing_temp']
    
    if all(col in df.columns for col in ['oil_pressure', 'coolant_pressure']):
        df['oil_coolant_pressure_ratio'] = df['oil_pressure'] / (df['coolant_pressure'] + 1e-6)
    
    if all(col in df.columns for col in ['engine_vibration', 'crankshaft_vibration']):
        df['vibration_composite'] = (df['engine_vibration'] + df['crankshaft_vibration']) / 2
    
    if all(col in df.columns for col in ['oil_temperature', 'ferrous_debris', 'soot_in_oil']):
        df['wear_composite'] = (100 - df['oil_temperature']) * 0.5 + df['ferrous_debris'] * 0.3 + df['soot_in_oil'] * 0.2
    
    return df

def in_threshold(value, threshold):
    if isinstance(threshold, (tuple, list)):
        # List of ranges
        if all(isinstance(t, tuple) for t in threshold):
            return any(t[0] <= value <= t[1] for t in threshold)
        # Single range
        elif len(threshold) == 2 and all(isinstance(x, (int, float)) for x in threshold):
            return threshold[0] <= value <= threshold[1]
    # Single value
    return value >= threshold

def train_xgboost_model(data_path):
    """
    Train an optimized XGBoost model on the provided dataset.
    
    Args:
        data_path (str): Path to the CSV file containing the training data
        
    Returns:
        tuple: (trained_model, scaler, feature_importances, metrics, feature_list)
    """
    df = pd.read_csv(data_path)
    
    base_features = [
        'injector_pressure', 'oil_pressure', 'coolant_pressure',
        'oil_temperature', 'ferrous_debris', 'soot_in_oil',
        'cylinder_head_temp', 'exhaust_gas_temp', 'bearing_temp',
        'engine_vibration', 'knock_sensor', 'crankshaft_vibration',
        'mass_air_flow', 'oxygen_sensor', 'egr_flow'
    ]
    
    missing_features = [f for f in base_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
    
    if 'RUL' not in df.columns:
        raise ValueError("Target variable 'RUL' not found in dataset")
    
    df = create_derived_features(df)
    
    all_features = base_features + [col for col in df.columns if col not in base_features + ['RUL']]
    
    X = df[all_features]
    y = df['RUL']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    split_idx = int(0.8 * len(X_scaled))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='mae'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    feature_importances = pd.DataFrame({
        'Feature': all_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'train': {
            'mae': mean_absolute_error(y_train, y_pred_train),
            'mse': mean_squared_error(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'r2': r2_score(y_train, y_pred_train)
        },
        'test': {
            'mae': mean_absolute_error(y_test, y_pred_test),
            'mse': mean_squared_error(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'r2': r2_score(y_test, y_pred_test)
        }
    }
    
    print("\nTraining Metrics:")
    print(f"MAE: {metrics['train']['mae']:.2f}")
    print(f"RMSE: {metrics['train']['rmse']:.2f}")
    print(f"R²: {metrics['train']['r2']:.4f}")
    
    print("\nTest Metrics:")
    print(f"MAE: {metrics['test']['mae']:.2f}")
    print(f"RMSE: {metrics['test']['rmse']:.2f}")
    print(f"R²: {metrics['test']['r2']:.4f}")
    
    return model, scaler, feature_importances, metrics, all_features

if __name__ == "__main__":
    os.makedirs('model', exist_ok=True)
    
    print("Training optimized XGBoost model...")
    model, scaler, feature_importances, metrics, feature_list = train_xgboost_model('data/fluid_sensor_data.csv')
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    feature_importances.to_csv('model/feature_importances.csv', index=False)
    with open('model/feature_list.pkl', 'wb') as f:
        pickle.dump(feature_list, f)
    
    print("\nFeature Importances:")
    print(feature_importances)
    print("\nModel training complete and artifacts saved.")