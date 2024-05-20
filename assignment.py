import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

features = pd.read_csv('temps.csv')

print('The shape of the dataset is:', features.shape)

#statistics on the dataset

# One-hot encode the data 
features = pd.get_dummies(features)
#print(features.iloc[:,5:].head(5))

labels = np.array(features['actual'])

features= features.drop('actual', axis = 1)
feature_list = list(features.columns)

features = np.array(features)
#print(features)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

#train_features, test_features = train_test_split(features, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


baseline_preds = test_features[:, feature_list.index('average')]
print("----------------------------Actual Values------------------------------------")
print(test_labels)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels)

predictions = rf.predict(test_features)

print("----------------------------Predicted Values---------------------------------")
print(predictions)
errors = abs(predictions - test_labels)

#baseline_errors = abs(baseline_preds - test_labels)
#print('Average baseline error: ', round(np.mean(baseline_errors), 2))

#print(errors)
print('\nMean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('\nAccuracy:', round(accuracy, 2), '%.')
