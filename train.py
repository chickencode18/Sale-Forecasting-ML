import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

# 1. Load data
file_path = 'scanner_data.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# 2. Aggregate daily sales
daily_sales = data.groupby('Date')['Sales_Amount'].sum().reset_index()
daily_sales = daily_sales.sort_values('Date').reset_index(drop=True)

# 3. Create time-based features
daily_sales['Day_of_Week'] = daily_sales['Date'].dt.dayofweek
daily_sales['Day_of_Month'] = daily_sales['Date'].dt.day
daily_sales['Week_of_Year'] = daily_sales['Date'].dt.isocalendar().week

# 4. Create extended features
daily_sales['Prev_Day_Sales'] = daily_sales['Sales_Amount'].shift(1)
daily_sales['Rolling_Mean_7'] = daily_sales['Sales_Amount'].rolling(window=7).mean()
daily_sales['Rolling_Mean_14'] = daily_sales['Sales_Amount'].rolling(window=14).mean()
daily_sales['Rolling_Std_7'] = daily_sales['Sales_Amount'].rolling(window=7).std()
daily_sales['Growth_Rate'] = daily_sales['Sales_Amount'].pct_change().replace([np.inf, -np.inf], 0)
daily_sales['Day_of_Week_sin'] = np.sin(2 * np.pi * daily_sales['Day_of_Week'] / 7)
daily_sales['Day_of_Week_cos'] = np.cos(2 * np.pi * daily_sales['Day_of_Week'] / 7)
daily_sales.fillna(0, inplace=True)

# 5. Create X (features) and y (target)
x = daily_sales[[
    'Day_of_Week_sin', 'Day_of_Week_cos',
    'Day_of_Month', 'Week_of_Year',
    'Prev_Day_Sales', 'Rolling_Mean_7', 'Rolling_Mean_14', 'Rolling_Std_7', 'Growth_Rate']]
y = daily_sales['Sales_Amount']

# 6. Split train/test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

# 7. Grid Search with RandomForest
cls = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        max_features='sqrt',
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    ))
])

# param_grid = {
#     'regressor__n_estimators': [200, 300, 500],
#     'regressor__max_depth': [20, 30, None],
#     'regressor__min_samples_split': [2, 5],
#     'regressor__min_samples_leaf': [1, 2],
#     'regressor__max_features': ['sqrt', 'log2']
# }

# grid_search = GridSearchCV(estimator=cls, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
# grid_search.fit(x_train, y_train)

# print("Best Parameters:", grid_search.best_params_)
cls.fit(x_train, y_train)

joblib.dump(cls, 'model.pkl')

y_pred = cls.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'MAPE: {mape*100:.2f}%')
print(f'R2 Score: {r2:.4f}')


plt.figure(figsize=(12,6))
plt.plot(daily_sales['Date'].iloc[-len(y_test):], y_test.values, label='Actual', marker='o')
plt.plot(daily_sales['Date'].iloc[-len(y_test):], y_pred, label='Predicted', marker='x')
plt.xlabel('Date')
plt.ylabel('Sales Amount')
plt.title('Comparison of Actual vs Predicted Daily Sales (Random Forest + GridSearchCV)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
