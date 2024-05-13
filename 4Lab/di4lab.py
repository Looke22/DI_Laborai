
#region BASIC SETUP
import pandas as pd

# Load data
melb_data = pd.read_csv('input/Melbourne_housing_FULL.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Drop rows that have missing 'Price' values
melb_data = melb_data.dropna(subset=['Price'])
#endregion

#region REMOVE ANOMALIES AND USELESS COLUMNS
#
# List of columns to be removed because of uselessness
columns_to_remove = ['Address', 'Postcode']  # Add other column names

# List of columns to be removed because they are not processed
#columns_to_remove_1 = ['Date', 'CouncilArea', 'Regionname']
columns_to_remove_1 = ['Date', 'CouncilArea', 'Regionname']

# Drop the columns from melb_candidate_predictors
melb_data = melb_data.drop(columns_to_remove, axis=1)
melb_data = melb_data.drop(columns_to_remove_1, axis=1)
#Lowercase
melb_data['Suburb'] = melb_data['Suburb'].str.lower()
print(melb_data)
#endregion

#region CATEGORIZE SUBURBS
import pandas as pd

# Group by 'Suburb' and calculate average price
suburb_avg_price = melb_data.groupby('Suburb')['Price'].mean()

# Define thresholds for categorization
affordable_threshold = suburb_avg_price.quantile(0.25)
mid_range_threshold = suburb_avg_price.quantile(0.5)
upscale_threshold = suburb_avg_price.quantile(0.75)

# Categorize suburbs based on average price
def categorize_suburb(price):
    if price < affordable_threshold:
        return 'Affordable'
    elif affordable_threshold <= price < mid_range_threshold:
        return 'Mid-Range'
    elif mid_range_threshold <= price < upscale_threshold:
        return 'Upscale'
    elif upscale_threshold <= price:
        return 'Luxury'

# Apply categorization function to each suburb
suburb_categories = suburb_avg_price.apply(categorize_suburb)

# Print the categories
print(suburb_categories)

#endregion

#region MAP SUBURB CATEGORIES
# Map suburb categories to the 'Suburb' column in the dataset
melb_data['Suburb'] = melb_data['Suburb'].map(suburb_categories)

# Print the updated dataset
print(melb_data)
#endregion

#region PREPARE FOR TRAINING
melb_target = melb_data.Price
melb_candidate_predictors = melb_data.drop(['Price'], axis=1)

# For the sake of keeping it simple, we'll use only numeric predictors.
melb_numeric_predictors = melb_candidate_predictors.select_dtypes(exclude=['object'])
#endregion

#region ONE-HOT ENCODING
melb_one_hot_predictors = pd.get_dummies(melb_candidate_predictors)
#endregion

#region SPLIT TRAIN AND TEST
PREDICTORS = melb_one_hot_predictors
X_train, X_test, y_train, y_test = train_test_split(PREDICTORS,
                                                    melb_target,
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    random_state=0)

def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
#endregion

#region TEST COLUMN DROP AND IMPUT
cols_with_missing = [col for col in X_train.columns
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
#endregion

#region IMPORTS AND IMPUTATION
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)

#Use Imputed values since they provide better results
X_train = imputed_X_train
X_test = imputed_X_test
#endregion

#region LINEAR REGRESSION
# Linear Regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_pred = linear_reg.predict(X_test)
linear_mae = mean_absolute_error(y_test, linear_pred)
print("Linear Regression MAE:", linear_mae)
linear_mse = mean_squared_error(y_test, linear_pred)
print("Linear Regression MSE:", linear_mse)
#endregion


#region DECISION TREES
# Decision Tree
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)
dt_pred = decision_tree.predict(X_test)
dt_mae = mean_absolute_error(y_test, dt_pred)
print("Decision Tree MAE:", dt_mae)
dt_mse = mean_squared_error(y_test, dt_pred)
print("Decision Tree MSE:", dt_mse)
#endregion

#region RNN
# Recurrent Neural Networks (RNN) using Keras
# Reshape data for LSTM input (assuming X_train and X_test are numpy arrays)
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build RNN model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[1])))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model
model.fit(X_train_rnn, y_train, epochs=200, batch_size=32, verbose=0)

# Predictions
rnn_pred = model.predict(X_test_rnn)
rnn_mae = mean_absolute_error(y_test, rnn_pred)
print("RNN MAE:", rnn_mae)
rnn_mse = mean_squared_error(y_test, rnn_pred)
print("RNN MSE:", rnn_mse)
#endregion

#region RANDOM FOREST
# Random Forest
random_forest = RandomForestRegressor(random_state=42)
random_forest.fit(X_train, y_train)
rf_pred = random_forest.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
print("Random Forest MAE:", rf_mae)
rf_mse = mean_squared_error(y_test, rf_pred)
print("Random Forest MSE:", rf_mse)
#endregion

#region COMPARISON
import matplotlib.pyplot as plt
import numpy as np

# Create a function to plot real vs predicted prices
def plot_comparison(real_prices, predicted_prices, model_name, ax):
    ax.scatter(real_prices, predicted_prices, alpha=0.35, label=model_name)

    ax.set_xlim([0, 5000000])
    ax.set_ylim([0, 5000000])
    ax.set_xlabel('Real Prices')
    ax.set_ylabel('Predicted Prices')
    ax.grid(True)

# Create a single figure for all plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

# Plot Linear Regression results
plot_comparison(y_test, linear_pred, 'Linear Regression', ax1)

# Plot Decision Tree results
plot_comparison(y_test, dt_pred, 'Decision Tree', ax1)

# Plot Random Forest results
plot_comparison(y_test, rf_pred, 'Random Forest', ax1)

# Plot RNN results
plot_comparison(y_test, rnn_pred.flatten(), 'Recurrent Neural Network', ax1)

# Add legend to the first plot
ax1.legend()
ax1.plot([0, np.max(y_test)], [0, np.max(y_test)], color='purple', linestyle='--', label='45-degree line')

# Plot the comparison for prices ranging from 0 to 2 million
plot_comparison(y_test, linear_pred, 'Linear Regression', ax2)
plot_comparison(y_test, dt_pred, 'Decision Tree', ax2)
plot_comparison(y_test, rf_pred, 'Random Forest', ax2)
plot_comparison(y_test, rnn_pred.flatten(), 'Recurrent Neural Network', ax2)

ax2.set_xlim([250000, 2000000])
ax2.set_ylim([0, 2500000])
ax2.set_xlabel('Real Prices')
ax2.set_ylabel('Predicted Prices')
ax2.grid(True)

# Add legend to the second plot
ax2.legend()
ax2.plot([0, np.max(y_test)], [0, np.max(y_test)], color='purple', linestyle='--', label='45-degree line')

plt.suptitle('Comparison of Predictive Models')
plt.show()
#endregion