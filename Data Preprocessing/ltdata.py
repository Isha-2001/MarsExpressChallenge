# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:52:55 2024

@author: DA
"""

import pandas as pd
import numpy as np

# Define file paths
filepath1 = "C:\\Users\\DA\\Desktop\\data\\train_set/context--2008-08-22_2010-07-10--ltdata.csv"
filepath2 = "C:\\Users\\DA\\Desktop\\data\\train_set/context--2010-07-10_2012-05-27--ltdata.csv"
filepath3 = "C:\\Users\\DA\\Desktop\\data\\train_set/context--2012-05-27_2014-04-14--ltdata.csv"
filepath4 = r"C:\Users\DA\Desktop\data\test_set\context--2014-04-14_2016-03-01--ltdata.csv"

# Load the data
ltdata1 = pd.read_csv(filepath1)
ltdata2 = pd.read_csv(filepath2)
ltdata3 = pd.read_csv(filepath3)
ltdata4 = pd.read_csv(filepath4)

# Concatenate data and process
ltdata = pd.concat([ltdata1, ltdata2, ltdata3], ignore_index=True)

ltdata['ut_ms'] = pd.to_datetime(ltdata['ut_ms'], unit='ms')
ltdata = ltdata.set_index('ut_ms')
hourly_data = ltdata.resample('1H').ffill()
hourly_data.to_csv('C:\\Users\\DA\\Desktop\\data\\resample\\ltdata_train.csv')

# Process the test set similarly

ltdata4['ut_ms'] = pd.to_datetime(ltdata4['ut_ms'], unit='ms')
ltdata4 = ltdata4.set_index('ut_ms')
hourly_data_test = ltdata4.resample('1H').ffill()
hourly_data_test.to_csv('C:\\Users\\DA\\Desktop\\data\\resample\\ltdata_test.csv')



import matplotlib.pyplot as plt
import seaborn as sns

def eda_for_file(file_path):
    df = pd.read_csv(file_path)
    
    # Print basic info and missing values
    print(f"Basic Info for {file_path}:")
    print(df.info())
    print("\nMissing values:\n", df.isnull().sum())
    
    # Handle infinite and missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Convert data to numeric for correlation calculation
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Statistical summary
    print("\nStatistical Summary:\n", df.describe())
    
    # Plot histograms for numerical features
    df.hist(figsize=(20, 15), bins=50)
    plt.suptitle('Histograms of Numerical Features')
    plt.show()
    
    # Correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

# Perform EDA for both training and testing datasets
train_file_path = 'C:\\Users\\DA\\Desktop\\data\\resample\\ltdata_train.csv'
test_file_path = 'C:\\Users\\DA\\Desktop\\data\\resample\\ltdata_test.csv'

eda_for_file(train_file_path)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def eda_for_file(file_path):
    df = pd.read_csv(file_path)
    
    # Handle infinite and missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Convert data to numeric for correlation calculation and exclude non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Plot histograms for numerical features
    df_numeric.hist(figsize=(20, 15), bins=50)
    plt.suptitle('Histograms of Numerical Features')
    plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

