# %%
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
from scipy.stats import uniform
from sklearn.metrics import mean_squared_error

# %%
features_df = pd.read_csv("mergedTrainNew.csv", parse_dates=["timestamp"], index_col="timestamp")
target_df = pd.read_csv("power_train.csv", parse_dates=["timestamp"], index_col="timestamp")
# %%
# Drop 'sunmars_km' before splitting the data
features_df = features_df.drop('sunmars_km', axis=1)
#%%
# Split the data into train and test sets. The test set will be used for the final evaluation
X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.2, random_state=42)
# %%
# MICE imputation to handle missing values
imputer = IterativeImputer(random_state=42)
# Impute missing values 
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Convert the imputed data back to DataFrames
X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)

# %%
# Train an Extra Trees model on the training data
feature_importances = pd.DataFrame(index=X_train_imputed.columns)

count = 1

for column in y_train.columns:
    model = ExtraTreesRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_imputed, y_train[column])
    feature_importances[column] = model.feature_importances_
    print("Done with " + str(count))
    count += 1

# Sort the feature importances in descending order for each column
sorted_feature_importances = feature_importances.apply(lambda x: x.sort_values(ascending=False), axis=1)
# Export the sorted feature importances to a CSV file
sorted_feature_importances.to_csv('feature_importances.csv')
# %%
# Also find the most common features in the top 10 across all the power lines
# Get the top 10 features for each power line
top_10_features = feature_importances.apply(lambda x: x.nlargest(10).index, axis=0)

# Create a flattened list of all top 10 features
all_top_10_features = top_10_features.values.flatten().tolist()

# Count the occurrences of each feature in the top 10 lists
feature_counts = pd.Series(all_top_10_features).value_counts()

# Sort the feature counts in descending order
sorted_feature_counts = feature_counts.sort_values(ascending=False)

# Print the most common features in the top 10 across all columns
print("Most common features in the top 10 across all columns:")
print(sorted_feature_counts)
sorted_feature_counts = sorted_feature_counts.reset_index()
sorted_feature_counts.columns = ['Feature', 'Count']
# Ploting
plt.figure(figsize=(10, 6))
plt.bar(sorted_feature_counts['Feature'], sorted_feature_counts['Count'])
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Count')
plt.title('Most Common Features in the Top 10 Across All Columns')
plt.show()
# %%
# Display the relative importance for each power line
#print("Relative Feature Importances:")
#print(sorted_feature_importances)

# Get the top 20 features for each power line
top_features = feature_importances.apply(lambda x: x.nlargest(20).index, axis=0)
print("\nTop 20 Features for each Power Line:")
print(top_features)
# %%
# Define the parameter grid for random search for XGBoost and ExtraTrees
xgboost_param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [6, 8, 10, 12],
    'learning_rate': uniform(0.01, 0.15),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.5, 0.5)
}

extra_trees_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [6, 8, 10, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Train XGBoost models and ExtraTree Models using the top 20 features for each power line with random search
xgb_models = {}
et_models = {}
xgb_best_params = {}
et_best_params = {}
predictions_dict = {'Power Line': [], 'XGBoost': [], 'ExtraTrees': [], 'Ensembled': []}

for column in y_train.columns:
    X_train_top = X_train_imputed[top_features[column]]
    y_train_column = y_train[column]

    # Create an XGBRegressor object
    xgb_model = XGBRegressor(random_state=42)

    # Perform random search for XGBoost
    xgb_random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=xgboost_param_grid,
        n_iter=35,
        scoring='neg_mean_squared_error',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    xgb_random_search.fit(X_train_top, y_train_column)

    # Get the best parameters and model for XGBoost
    xgb_best_params[column] = xgb_random_search.best_params_
    xgb_models[column] = xgb_random_search.best_estimator_

    # Create an ExtraTreesRegressor object
    et_model = ExtraTreesRegressor(random_state=42)

    # Perform random search for ExtraTrees
    et_random_search = RandomizedSearchCV(
        estimator=et_model,
        param_distributions=extra_trees_param_grid,
        n_iter=35,
        scoring='neg_mean_squared_error',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    et_random_search.fit(X_train_top, y_train_column)

    # Get the best parameters and model for ExtraTrees
    et_best_params[column] = et_random_search.best_params_
    et_models[column] = et_random_search.best_estimator_

    # Retrain the best XGBoost model on the entire training set
    xgb_models[column].fit(X_train_top, y_train_column)

    # Retrain the best ExtraTrees model on the entire training set
    et_models[column].fit(X_train_top, y_train_column)

    # Predict on the test set using the best models
    y_pred_test_xgb = xgb_models[column].predict(X_test_imputed[top_features[column]])
    y_pred_test_et = et_models[column].predict(X_test_imputed[top_features[column]])
    y_pred_test_ensemble = (y_pred_test_xgb + y_pred_test_et) / 2

    # Calculate RMSE for the test set predictions
    rmse_test_xgb = mean_squared_error(y_test[column], y_pred_test_xgb, squared=False)
    rmse_test_et = mean_squared_error(y_test[column], y_pred_test_et, squared=False)
    rmse_test_ensemble = mean_squared_error(y_test[column], y_pred_test_ensemble, squared=False)

    # Append the power line and RMSE values to the predictions dictionary
    predictions_dict['Power Line'].append(column)
    predictions_dict['XGBoost'].append(rmse_test_xgb)
    predictions_dict['ExtraTrees'].append(rmse_test_et)
    predictions_dict['Ensembled'].append(rmse_test_ensemble)

# Convert the best parameters dictionaries to DataFrames
xgb_best_params_df = pd.DataFrame.from_dict(xgb_best_params, orient='index')
et_best_params_df = pd.DataFrame.from_dict(et_best_params, orient='index')

# Export the DataFrames to CSV files
xgb_best_params_df.to_csv('xgb_best_params.csv')
et_best_params_df.to_csv('et_best_params.csv')

# Convert the predictions dictionary to a DataFrame
predictions_df = pd.DataFrame(predictions_dict)

# Print the mean RMSE across all power lines for XGBoost, ExtraTrees, and the Ensembled Model
mean_rmse_xgb = predictions_df['XGBoost'].mean()
mean_rmse_et = predictions_df['ExtraTrees'].mean()
mean_rmse_ensemble = predictions_df['Ensembled'].mean()
print(f"XGBoost Mean RMSE: {mean_rmse_xgb:.4f}")
print(f"ExtraTrees Mean RMSE: {mean_rmse_et:.4f}")
print(f"Ensembled Mean RMSE: {mean_rmse_ensemble:.4f}")
# Export the predictions DataFrame to a CSV file
# %%
predictions_df.to_csv('model_predictions.csv', index=False)