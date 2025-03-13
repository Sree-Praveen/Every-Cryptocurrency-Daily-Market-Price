# Every-Cryptocurrency-Daily-Market-Price
Team - 19


# Project Overview

## Target Audience

### Institutional Investors
Hedge funds and asset management firms can use predictions to inform trading strategies and portfolio construction.

### Retail Investors
Individual traders can leverage forecasts to make informed decisions about buying or selling cryptocurrencies.

### Cryptocurrency Exchanges
Platforms like Binance and Coinbase can use volatility predictions to improve liquidity management.

### Fintech Companies
Firms offering cryptocurrency-related services can integrate predictive analytics into their platforms to enhance customer engagement.

---

The cryptocurrency market is characterized by high volatility, rapid shifts in trends, and complex interdependencies between assets. These characteristics make it an ideal candidate for advanced predictive analytics using deep learning techniques. This business case outlines the development of a deep learning-based predictive analytics system leveraging the Kaggle cryptocurrency dataset. 

The system aims to address critical business questions, including:
- Price prediction
- Risk management
- Market regime classification

By integrating state-of-the-art deep learning methods such as LSTM networks, CNNs, and hybrid architectures, this project seeks to provide actionable insights for investors, exchanges, financial analysts, and fintech companies.

## Key Features Used for Prediction
- High Value
- Close Ratio
- High
- Low
- Close
- Spread
- Volume

---

## Required Libraries
- **Pandas**: For data manipulation and analysis.
- **Numpy**: For numerical operations and handling arrays.
- **Scikit-learn**: For building and evaluating machine learning models.
- **Matplotlib.pyplot**: For creating visualizations like charts and plots.
- **Seaborn**: For statistical data visualization, providing an interface for creating complex visualizations easily.
- **Shap**: For model explainability, to analyze how individual features contribute to model predictions.

---

## Methodology
- Addressed missing values, outliers, and inconsistencies within the dataset.
- Numerical features are scaled using the `StandardScaler` to ensure that all features contribute equally to the model.

---

## Dataset Details
### Main Dataset:
[All Crypto Currencies - Kaggle](https://www.kaggle.com/datasets/jessevent/all-crypto-currencies/data)
- **Date Range Used**: 2015-01-01 to 2018-12-31

### Additional Dataset:
[Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/series/T10YIE)
- **Date Range Used**: 2015-01-01 to 2018-12-31

---

## Model Selection and Training
An **LSTM-based neural network** is used for binary classification, where the model predicts whether the closing price will increase (`1`) or decrease (`0`) on the next day.
