import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from train_model import train_xgboost_model, create_derived_features
import seaborn as sns

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

input_features = [
    'injector_pressure', 'oil_pressure', 'coolant_pressure',
    'oil_temperature', 'ferrous_debris', 'soot_in_oil',
    'cylinder_head_temp', 'exhaust_gas_temp', 'bearing_temp',
    'engine_vibration', 'knock_sensor', 'crankshaft_vibration',
    'mass_air_flow', 'oxygen_sensor', 'egr_flow'
]

st.title("Engine Health Monitoring Dashboard")

SENSOR_CATEGORIES = {
    "Wear & Degradation": {
        "ferrous_debris": {
            "min": 0, "max": 100, "unit": "um",
            "warning": (20, 99),  
            "critical": 100 
        },
        "soot_in_oil": {
            "min": 0, "max": 140, "unit": "mg/L",
            "warning": 10000,  
            "critical": 20000,  
        }
    },
    "Temperature & Thermal": {
        "cylinder_head_temp": {
            "min": 70, "max": 130, "unit": "°C",
            "warning": (110, 129),  
            "critical": 130  
        },
        "exhaust_gas_temp": {
            "min": 250, "max": 850, "unit": "°C",
            "warning": (650, 849),  
            "critical": 850  
        },
        "bearing_temp": {
            "min": 30, "max": 110, "unit": "°C",
            "warning": (85, 109), 
            "critical": 110  
        }
    },
    "Vibration & Mechanical": {
        "engine_vibration": {
            "min": 0, "max": 10, "unit": "gravity(g)",
            "warning": (5, 9),  
            "critical": 10  
        },
        "knock_sensor": {
            "min": 0, "max": 100, "unit": "%",
            "warning": 70,  
            "critical": 85  
        },
        "crankshaft_vibration": {
            "min": 200, "max": 1000, "unit": "ohms",
            "warning": [(200, 250), (900, 1000)], 
            "critical": [(0, 200), (1000, float('inf'))]  
        }
    },
    "Fluid & Pressure": {
        "oil_temperature": {
            "min": 0, "max": 120, "unit": "°C",
            "warning": 15000, 
            "critical": 15000, 
        },
        "injector_pressure": {
            "min": 1, "max": 20, "unit": "MPa",
            "warning": 10,  
            "critical": 20  
        },
        "oil_pressure": {
            "min": 0, "max": 0.5, "unit": "MPa",
            "warning": (0.2, 0.29),  
            "critical": 0.3
        },
        "coolant_pressure": {
            "min": 0, "max": 0.15, "unit": "MPa",
            "warning": (0.10, 0.15),  
            "critical": [(0, 0.09), (0.25, float('inf'))]  
        }
    },
    "Air & Combustion": {
        "mass_air_flow": {
            "min": 0.2, "max": 10, "unit": "m/s",
            "warning": 3.3, 
            "critical": 4.4  
        },
        "oxygen_sensor": {
            "min": 0, "max": 1.1, "unit": "lambda",
            "warning": (0.90, 0.95),  
            "critical": 0.89  
        },
        "egr_flow": {
            "min": 0, "max": 20, "unit": "%mass",
            "warning": (15, 19),  
            "critical": 20  
        }
    }
}

st.subheader("Upload Sensor Data")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
data_path = 'data/fluid_sensor_data.csv'

if uploaded_file is not None:
    temp_path = "data/temp_upload.csv"
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    data_path = temp_path

try:
    df = pd.read_csv(data_path)
    missing_columns = [col for col in input_features if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.stop()
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                model, scaler, feature_importances, metrics, feature_list = train_xgboost_model(data_path)
                os.makedirs('model', exist_ok=True)
                joblib.dump(model, 'model/model.joblib', compress=3)
                joblib.dump(scaler, 'model/scaler.joblib', compress=3)
                feature_importances.to_csv('model/feature_importances.csv', index=False)
                joblib.dump(feature_list, 'model/feature_list.joblib', compress=3)
                st.success("Model trained successfully!")
                
                st.write("Training Metrics:")
                st.write(f"MAE: {metrics['train']['mae']:.2f}")
                st.write(f"RMSE: {metrics['train']['rmse']:.2f}")
                st.write(f"R²: {metrics['train']['r2']:.4f}")
                
                st.write("Test Metrics:")
                st.write(f"MAE: {metrics['test']['mae']:.2f}")
                st.write(f"RMSE: {metrics['test']['rmse']:.2f}")
                st.write(f"R²: {metrics['test']['r2']:.4f}")
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                st.stop()
    else:
        try:
            model = joblib.load('model/model.joblib')
            scaler = joblib.load('model/scaler.joblib')
            feature_list = joblib.load('model/feature_list.joblib')
        except:
            st.warning("Please train the model first!")
            st.stop()
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Component Health", "Sensor Details", "System Overview", "Manual RUL Prediction"])

with tab1:
    st.subheader("Component Health Status")
    
    fig, axes = plt.subplots(len(SENSOR_CATEGORIES), 1, figsize=(12, 4*len(SENSOR_CATEGORIES)))
    fig.suptitle("Component Health Status", fontsize=16)
    
    for idx, (category, sensors) in enumerate(SENSOR_CATEGORIES.items()):
        ax = axes[idx]
        ax.set_title(category)
        
        component_values = []
        for sensor, specs in sensors.items():
            if sensor in df.columns:
                value = df[sensor].iloc[-1]
                normalized_value = (value - specs["min"]) / (specs["max"] - specs["min"]) * 100
                component_values.append(normalized_value)
        
        component_rul = 100 - np.mean(component_values)
        component_rul = np.clip(component_rul, 0, 100)
        
        ax.barh([0], [20], color='red', left=0, label='Critical')
        ax.barh([0], [30], color='yellow', left=20, label='Warning')
        ax.barh([0], [50], color='green', left=50, label='Good')
        ax.scatter([component_rul], [0], color='black', s=200, zorder=5, label='Component RUL')
        
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel('RUL')
        ax.legend()
        
        if component_rul <= 20:
            st.error(f"{category}: RUL is critically low! Immediate maintenance required.")
        elif component_rul <= 50:
            st.warning(f"{category}: RUL is getting low. Plan maintenance soon.")
        else:
            st.success(f"{category}: Good condition.")
    
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    st.subheader("Detailed Sensor Information")
    
    for category, sensors in SENSOR_CATEGORIES.items():
        st.markdown(f"### {category}")
        cols = st.columns(len(sensors))
        
        for idx, (sensor, specs) in enumerate(sensors.items()):
            with cols[idx]:
                if sensor in df.columns:
                    value = df[sensor].iloc[-1]
                    # Calculate RUL for this sensor
                    normalized_value = (value - specs["min"]) / (specs["max"] - specs["min"]) * 100
                    sensor_rul = 100 - normalized_value
                    sensor_rul = np.clip(sensor_rul, 0, 100)
                    
                    st.metric(
                        label=sensor.replace('_', ' ').title(),
                        value=f"{value:.1f}{specs['unit']}",
                        delta=f"RUL: {sensor_rul:.1f}%"
                    )
                    
                    # Show warning if value exceeds thresholds
                    if in_threshold(value, specs["critical"]):
                        st.error("Critical Level!")
                    elif in_threshold(value, specs["warning"]):
                        st.warning("Warning Level!")
                    else:
                        st.success("Normal Level")
                    
                    # Plot historical data with RUL
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4), height_ratios=[2, 1])
                    
                    # Plot sensor values
                    ax1.plot(df[sensor].tail(20), color='blue', label='Sensor Value')
                    # Plot warning threshold(s)
                    warning = specs["warning"]
                    if isinstance(warning, (tuple, list)):
                        # List of ranges
                        if all(isinstance(t, tuple) for t in warning):
                            for t in warning:
                                ax1.axhline(y=t[0], color='yellow', linestyle='--', alpha=0.5, label='Warning')
                                ax1.axhline(y=t[1], color='yellow', linestyle='--', alpha=0.5)
                        # Single range
                        elif len(warning) == 2 and all(isinstance(x, (int, float)) for x in warning):
                            ax1.axhline(y=warning[0], color='yellow', linestyle='--', alpha=0.5, label='Warning')
                            ax1.axhline(y=warning[1], color='yellow', linestyle='--', alpha=0.5)
                        else:
                            pass  # skip if not a valid range
                    else:
                        ax1.axhline(y=warning, color='yellow', linestyle='--', alpha=0.5, label='Warning')

                    # Plot critical threshold(s)
                    critical = specs["critical"]
                    if isinstance(critical, (tuple, list)):
                        if all(isinstance(t, tuple) for t in critical):
                            for t in critical:
                                ax1.axhline(y=t[0], color='red', linestyle='--', alpha=0.5, label='Critical')
                                ax1.axhline(y=t[1], color='red', linestyle='--', alpha=0.5)
                        elif len(critical) == 2 and all(isinstance(x, (int, float)) for x in critical):
                            ax1.axhline(y=critical[0], color='red', linestyle='--', alpha=0.5, label='Critical')
                            ax1.axhline(y=critical[1], color='red', linestyle='--', alpha=0.5)
                        else:
                            pass
                    else:
                        ax1.axhline(y=critical, color='red', linestyle='--', alpha=0.5, label='Critical')

                    ax1.set_title(f"{sensor.replace('_', ' ').title()}")
                    ax1.legend(loc='upper right')
                    
                    # Calculate and plot RUL
                    sensor_values = df[sensor].tail(20)
                    rul_values = 100 - ((sensor_values - specs["min"]) / (specs["max"] - specs["min"]) * 100)
                    rul_values = rul_values.clip(0, 100)
                    
                    # Plot RUL
                    ax2.plot(rul_values, color='green', label='RUL')
                    ax2.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Critical')
                    ax2.axhline(y=50, color='yellow', linestyle='--', alpha=0.5, label='Warning')
                    ax2.set_ylim(0, 100)
                    ax2.set_ylabel('RUL %')
                    ax2.legend(loc='upper right')
                    
                    plt.tight_layout()
                    st.pyplot(fig)

with tab3:
    st.subheader("System Overview")
    
    st.subheader("Critical Alerts")
    for category, sensors in SENSOR_CATEGORIES.items():
        for sensor, specs in sensors.items():
            if sensor in df.columns:
                value = df[sensor].iloc[-1]
                if in_threshold(value, specs["critical"]):
                    st.error(f"CRITICAL: {sensor.replace('_', ' ').title()} in {category} is at {value:.1f}{specs['unit']}")
                elif in_threshold(value, specs["warning"]):
                    st.warning(f"WARNING: {sensor.replace('_', ' ').title()} in {category} is at {value:.1f}{specs['unit']}")

with tab4:
    st.subheader("Manual RUL Prediction")
    st.markdown("Enter sensor values to predict RUL for each component")
    
    for category, sensors in SENSOR_CATEGORIES.items():
        st.markdown(f"### {category}")
        cols = st.columns(len(sensors))
        
        user_inputs = {}
        
        for idx, (sensor, specs) in enumerate(sensors.items()):
            with cols[idx]:
                st.markdown(f"**{sensor.replace('_', ' ').title()}**")
                value = st.number_input(
                    f"Value ({specs['unit']})",
                    min_value=float(specs['min']),
                    max_value=float(specs['max']),
                    value=float(specs['min']),
                    key=sensor
                )
                user_inputs[sensor] = value
                
                if in_threshold(value, specs["critical"]):
                    st.error("Critical Level!")
                elif in_threshold(value, specs["warning"]):
                    st.warning("Warning Level!")
                else:
                    st.success("Normal Level")
        
        if st.button(f"Predict RUL for {category}", key=f"predict_{category}"):
            normalized_values = []
            for sensor, value in user_inputs.items():
                specs = SENSOR_CATEGORIES[category][sensor]
                normalized_value = (value - specs["min"]) / (specs["max"] - specs["min"]) * 100
                normalized_values.append(normalized_value)
            
            category_rul = 100 - np.mean(normalized_values)
            category_rul = np.clip(category_rul, 0, 100)
            
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.barh([0], [20], color='red', left=0, label='Critical')
            ax.barh([0], [30], color='yellow', left=20, label='Warning')
            ax.barh([0], [50], color='green', left=50, label='Good')
            ax.scatter([category_rul], [0], color='black', s=200, zorder=5, label='Predicted RUL')
            
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xlabel('RUL')
            ax.legend()
            plt.title(f'{category} RUL Prediction')
            st.pyplot(fig)

            # Display RUL value and status
            st.markdown(f"### Predicted RUL: **{category_rul:.2f}%**")
            if category_rul <= 20:
                st.error("CRITICAL: Immediate maintenance required!")
            elif category_rul <= 50:
                st.warning("WARNING: Plan maintenance soon.")
            else:
                st.success("GOOD: Component is in good condition.")
            
            st.markdown("#### Component-wise Analysis:")
            for sensor, value in user_inputs.items():
                specs = SENSOR_CATEGORIES[category][sensor]
                normalized_value = (value - specs["min"]) / (specs["max"] - specs["min"]) * 100
                component_rul = 100 - normalized_value
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**{sensor.replace('_', ' ').title()}**")
                with col2:
                    st.progress(component_rul/100)
                    st.markdown(f"RUL: {component_rul:.1f}%")