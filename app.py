import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

malbourne_file_path = './melb_data.csv'

# read csv
malbourne_data = pd.read_csv(malbourne_file_path)

# remove empty lines
cleanData = malbourne_data.dropna(axis=0)

# select column & select one feature
y = cleanData.Price

# select many feature
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = cleanData[melbourne_features]

# summary of the data example count, min, max ...
temp = X.describe()

# top 5 colums
temp = X.head()

melbourne_model = DecisionTreeRegressor()
temp = melbourne_model.fit(X, y)

# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# print(melbourne_model.predict(X.head()))

# print(temp)

predicted_home_prices = melbourne_model.predict(X)
temp = mean_absolute_error(y, predicted_home_prices)

# print(temp)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)

val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))