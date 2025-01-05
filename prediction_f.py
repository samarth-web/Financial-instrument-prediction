

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import streamlit as st
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

def prediction(symbol):
    if symbol == "":
        symbol = 'AAPL'
    data = yf.download(symbol, start='2023-01-01', end='2025-01-01')
    data['hi',symbol] = data['Close'].rolling(window=10).mean()

    data['SMA10'] = data['Close'].rolling(window=10, min_periods=1).mean()
    data['SMA50'] = data['Close'].rolling(window=50, min_periods=1).mean()

    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    data['RSI'] = 100 - (100 / (1 + avg_gain / avg_loss))

    data['Price_Change'] = data['Close'].diff()
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data['EMA10',symbol] = data['Close'].ewm(span=10, adjust=False).mean()
    data['BB_upper',symbol] = data['Close'].rolling(20).mean() + 2 * data['Close'].rolling(20).std()
    data['BB_lower',symbol] = data['Close'].rolling(20).mean() - 2 * data['Close'].rolling(20).std()
    data['MACD',symbol] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()




    data.fillna(method='ffill', inplace=True)


    print(data['Target'])
    X = data[['SMA10', 'SMA50', 'RSI', 'Price_Change','EMA10','BB_upper','BB_lower','MACD']]
    y = data['Target']


    X = X.copy()
    X.fillna(method='ffill', inplace=True)
    X.fillna(0, inplace=True)
    X = X.iloc[:-1]
    y = y.iloc[:-1]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    pipeline = Pipeline([
        ('scaler', StandardScaler()),  
        ('classifier', RandomForestClassifier(random_state=42))
    ])


    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }


    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,  
        scoring='accuracy',  
        verbose=1, 
        n_jobs=-1 
    )

    grid_search.fit(X_train, y_train.values.ravel())


    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)


    y_pred = grid_search.best_estimator_.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    importance = grid_search.best_estimator_.feature_importances_
    features = X.columns


    indices = np.argsort(importance)[::-1]  
    sorted_features = features[indices]    

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(features)), importance[indices], align="center")
    plt.xticks(range(len(features)), sorted_features, rotation=45, ha="right")  
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    


    test_data = X_test.copy()


    test_data['Signal'] = y_pred


    test_data = test_data.reindex(data.index)  
    test_data['Close'] = data['Close']  

    test_data.fillna(method='ffill', inplace=True)  
    test_data.fillna(method='bfill', inplace=True) 


    test_data['Signal'] = test_data['Signal'].shift(1)

    test_data['Daily_Returns'] = test_data['Close'].pct_change()

    test_data['Strategy_Returns'] = test_data['Signal'] * test_data['Daily_Returns']


    test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    test_data.fillna(0, inplace=True)


    test_data['Cumulative_Strategy'] = (1 + test_data['Strategy_Returns']).cumprod() - 1


    test_data['Normalized_Close'] = data['Close'] / data['Close'].iloc[0]
    test_data['Cumulative_Actual'] = test_data['Normalized_Close'] - 1


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(test_data.index, test_data['Cumulative_Strategy'], label='Model Strategy', color='blue')
    ax.plot(test_data.index, test_data['Cumulative_Actual'], label='Actual Returns', color='orange')
    ax.legend()
    ax.set_title(f"Trading Strategy Performance vs. Actual Returns ({symbol})")

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

    mean_return = test_data['Strategy_Returns'].mean()
    std_dev = test_data['Strategy_Returns'].std()
    
    if std_dev != 0:
        sharpe_ratio = mean_return / std_dev 
    else:
        sharpe_ratio = 0

    rolling_max = test_data['Cumulative_Strategy'].cummax()
    drawdown = rolling_max - test_data['Cumulative_Strategy']
    max_drawdown = drawdown.max()


    print(f"Performance Metrics ({symbol}):")
    print(f"- Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"- Max Drawdown: {max_drawdown:.2%}")

    sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    # print(test_data.index.equals(data.index))  # Should return True
    # print(test_data[['Close', 'Normalized_Close']].head())  # Check
    st.pyplot()

st.title('Financial Instrument Prediction')
user_input = st.text_input('Enter Instrument (e.g., symbol):')
symbol = user_input
st.set_option('deprecation.showPyplotGlobalUse', False)
prediction(symbol)