import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t

# Column names
column_names = [
    "Subject_identifier", "Jitter_%", "Jitter_Abs", "Jitter_RAP", "Jitter_PPQ5", "Jitter_DDP",
    "Shimmer_%","Shimmer_db", "Shimmer_APQ3", "Shimmer_APQ5", "Shimmer_APQ11", "Shimmer_DDA",
    "Harmonicity", "Harmonicity_NHR", "Harmonicity_HNR", "Pitch_Median", "Pitch_Mean",
    "Pitch_StdDev", "Pitch_Min", "Pitch_Max", "Pulse_Num", "Pulse_Periods", "Pulse_Mean",
    "Pulse_StdDev", "Voice_FractionUnvoiced", "Voice_NumVoiceBreaks", "Voice_DegreeVoiceBreaks",
    "UPDRS", "PD_Indicator"
]

# Load the dataset
data = pd.read_csv('po1_data.csv')
data.columns = column_names

# Drop rows and columns where all values are NaN
data.dropna(axis=0, how='all', inplace=True)
data.dropna(axis=1, how='all', inplace=True)

# Separate PD and healthy groups
pd_group = data[data['PD_Indicator'] == 1]
healthy_group = data[data['PD_Indicator'] == 0]

# Select features for analysis
selected_features = ["Jitter_%", "Jitter_RAP", "Jitter_PPQ5", "Jitter_DDP","Shimmer_APQ11",
    "Harmonicity_NHR","Pitch_StdDev","Voice_FractionUnvoiced", "Voice_NumVoiceBreaks", "Voice_DegreeVoiceBreaks"
]

# List to store statistically significant features
significant_features = []

# Data wrangling and analysis
for feature in selected_features:
    pd_values = pd_group[feature]
    healthy_values = healthy_group[feature]
    
    # Summary statistics
    pd_mean = np.mean(pd_values)
    healthy_mean = np.mean(healthy_values)
    pd_std = np.std(pd_values, ddof=1)
    healthy_std = np.std(healthy_values, ddof=1)
    
    # Z-test for PD group compared to healthy group
    z_stat = (pd_mean - healthy_mean) / np.sqrt(pd_std**2/len(pd_values) + healthy_std**2/len(healthy_values))
    p_value_z = 2 * (1 - norm.cdf(np.abs(z_stat)))  # Two-tailed test
    
    # T-test for PD group compared to healthy group
    pooled_std = np.sqrt(((len(pd_values) - 1) * pd_std**2 + (len(healthy_values) - 1) * healthy_std**2) / (len(pd_values) + len(healthy_values) - 2))
    t_stat = (pd_mean - healthy_mean) / (pooled_std * np.sqrt(1/len(pd_values) + 1/len(healthy_values)))
    df = len(pd_values) + len(healthy_values) - 2  # degrees of freedom for t-test
    p_value_t = 2 * (1 - t.cdf(np.abs(t_stat), df=df))
    
    # Check for statistical significance
    alpha = 0.05  # thresold level
    if p_value_z < alpha or p_value_t < alpha:
        significant_features.append(feature)
    
    print(f'Feature: {feature}')
    print(f'PD Group Mean: {pd_mean:.2f}, Healthy Group Mean: {healthy_mean:.2f}')
    print(f'PD Group Standard Deviation: {pd_std:.2f}, Healthy Group Standard Deviation: {healthy_std:.2f}')
    print(f'Z-Statistic (PD vs Healthy): {z_stat:.2f}, P-Value (Z-test): {p_value_z:.4f}')
    print(f'T-Statistic (PD vs Healthy): {t_stat:.2f}, P-Value (T-test): {p_value_t:.4f}')
    print('='*50)

# Visualization
for feature in selected_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pd_group, x=feature, label='PD Group', color='blue', kde=True)
    sns.histplot(data=healthy_group, x=feature, label='Healthy Group', color='orange', kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

print("Statistically Significant Features:")
for feature in significant_features:
    print(feature)
