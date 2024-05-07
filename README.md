# MarsExpressChallenge
Course Code : DATA70202

Project Objectives
The primary objective of this project is to develop a machine learning model that accurately predicts the average hourly electric current for each of the 33 power lines in MEX, using the provided context data. The model's performance is evaluated using root mean squared error (RMSE), which was the chosen metric in the original competition. Accurate predictions of power consumption are crucial for optimising space operations and ensuring the longevity and safety of the spacecraft.

Approach
1. We recognised that each feature we had would have varying effects on the 33 different power lines, as the power lines were distinct from each other. Therefore, we decided to build 33 separate machine learning models, each with its own set of features. We came up with the following approach:
2. Reserve 20% of the data from Years 1 to 3, treating it as a test set.
3. Build 33 simple Extra Trees models on the training set without any hyperparameter tuning for each power line, with the sole purpose of extracting feature importances.
4. Extract the top 20 most important features for each power line.
5. Rebuild the 33 Extra Trees models with the features from step 2. Use hyperparameter tuning and 5-fold cross-validation to optimise performance on the validation set, measuring performance using the root mean squared error (RMSE) metric. Retrain the final model on both the training and validation set using the optimal set of hyperparameters.
6. Develop 33 XGBoost models, also using hyperparameter tuning and 5-fold cross-validation. Retrain the final model on both the training and validation set using the optimal set of hyperparameters.
7. For each power line, ensemble the results of its accompanying Extra Trees and XGBoost model using a simple mean to create a third, combined model.
8. Compare the performance of all three model types (Extra Trees, XGBoost, and ensembled) on the reserved test set, keeping the best performing model.
