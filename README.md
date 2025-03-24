# Every Cryptocurrency Daily Market Price
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
- **Date Range Used**: 2015-01-01 to 2018-11-30

### Additional Dataset:
[Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/series/T10YIE)
- **Date Range Used**: 2015-01-01 to 2018-11-30

[Snp500 VOO Data](https://www.nasdaq.com/market-activity/etf/voo/historical)
- **Date Range Used**: 2015-01-01 to 2018-11-30
---
![image](https://github.com/user-attachments/assets/248e1058-24ff-433c-92f7-6973c3d0ddda)

For the period from 2015-01-01 to 2018-11-30, the correlation between cryptocurrency market data, the S&P 500 index, and inflation data appears to be weak or even negative. While traditional financial markets, represented by the S&P 500, are influenced by macroeconomic factors like interest rates and inflation, the cryptocurrency market operates largely independently, driven by factors such as speculation, technological advancements, regulatory changes, and adoption trends. During this period, Bitcoin and other cryptocurrencies experienced a significant bull run (notably in 2017), while inflation and the S&P 500 followed more stable trends. The divergence suggests that cryptocurrencies did not behave as a traditional hedge against inflation or closely track stock market movements, reinforcing their position as an alternative asset class rather than a direct reflection of traditional economic indicators.

---
## Data Preprocessing

### Exploratory Data Analysis (EDA) on Cryptocurrency Data

- Checks for missing values.

- Detects outliers in key numerical features (open, high, low, close, volume, 
market) using box plots with a log scale for better visualization.

### Filtering and Aggregating Data for Analysis

- Filters dataset to include only records from 2015-01-01 to 2018-12-31.

- Calculates the average trading volume per cryptocurrency by grouping the 
data by symbol.

### Top 10 cryptocurrencies by average volume (2015-2018):

![image](https://github.com/user-attachments/assets/1d6d1943-6365-41b9-ab6e-f1e7f60a71a3)


### Outlier Treatment & Data Cleaning

- Creates an explicit copy of the filtered dataset.

- Applies Winsorization to numerical features (open, high, low, close, volume,
market) by capping extreme values at the 1st and 99th percentiles to reduce
the impact of outliers.

- Identifies and removes duplicate rows (based on date and symbol) to ensure
uniqueness.

- Performs a final missing value check to confirm data completeness.
  
---

## Feature Extraction

### Importance of Feature Extraction

- Enhances predictive model performance by deriving meaningful features 
from raw data.

- Technical Indicators Implemented:

- Simple Moving Average (SMA 7-day): Smooths price trends.

- Relative Strength Index (RSI 14-day): Measures momentum to identify 
overbought/oversold conditions.

- Exponential Moving Averages (EMA 12-day & 26-day): Captures short- 
and long-term trends.

- Moving Average Convergence Divergence (MACD): Identifies trend 
reversals.

- Bollinger Bands (20-day): Measures price volatility using standard 
deviation.

---

## Data Splitting

### Classifying Market Conditions:

- The function classifies market conditions (bullish, sideways, or bearish) based
on the percentage change in closing prices.

### Preparing Data for Time Series Cross-Validation:

- TimeSeriesSplit is used to perform time series cross-validation with 3 folds.

### Splitting Data for Training, Validation, and Testing:

- The dataset is split into training, validation, and test sets in each fold of the 
cross-validation.

- The test set is further divided into validation and test sets.

- The splits are stored in lists and then concatenated to create the final 
training, validation, and test datasets.

---

## Model Training, evaluation and optimization:

- Utilized LSTM model-> loaded and prepared the data, performed feature 
engineering, trained the LSTM model, and evaluated its performance.

- Calculated Percentage Price Changes, based on the computed percentage 
changes, the function assigns a market condition label (bullish, bearish, 
or neutral) using predefined thresholds. If neither condition is met, it defaults 
to neutral.

- Trained, scaled features using MinMaxScaler which This helps improve LSTM 
performance by ensuring that all input values are on a similar scale.

- Applied one hot encoding, Class weights are computed to address class 
imbalance by assigning higher weights to underrepresented classes.

- The model is designed with two Bidirectional LSTM layers to capture both 
forward and backward temporal dependencies. It 
includes BatchNormalization to stabilize and accelerate training, 
and Dropout layers for regularization, preventing overfitting.

- A Dense layer with ReLU activation enables non-linear feature learning, while 
the final softmax output layer provides multi-class classification.

![image](https://github.com/user-attachments/assets/6ff6c6e5-f3c3-497e-8edd-6600b76ab638)


- The architecture is optimized using the Adam optimizer and categorical cross-
entropy loss for better performance and generalization.

- Model is trained using training data, epoch=20, batch size=64, callbacks and 
class weights.

![image](https://github.com/user-attachments/assets/2681d45b-8bcd-4035-a229-840f52723bc5)

![image](https://github.com/user-attachments/assets/f5fdcfbf-fad2-4950-80eb-330e873e3c5e)

![image](https://github.com/user-attachments/assets/3d76bfe3-ad7f-4f5a-9712-93a427e4ee0d)


- The best model is loaded and evaluated on the test set, reporting the test 
accuracy. The training and validation accuracy and loss over epochs are 
plotted in two subplots to visualize model performance during training. The 
plots display how the model's accuracy and loss evolved for both training and
validation data. The test accuracy of the model is 0.8120.



![image](https://github.com/user-attachments/assets/f7771584-505e-4113-9515-d1df439285a0)


![image](https://github.com/user-attachments/assets/ccffea40-21f8-45eb-8ef5-38d1805dbfe7)


## Team

| Name   | Video           |
|--------|------------------|
| Eromosele Ikhalo | [Watch Video](https://youtu.be/uIHFIQuMeVg) |
| Sreelakshmi Praveen | [Watch Video](#) |
| Max Trunov  | [Watch Video](#) |
| Vishakha Nair | [Watch Video](https://drive.google.com/file/d/1Y0F6UGRktNU0Y9iZIUTkWJFzPELZYcnp/view?usp=drivesdk) |
| Lakshaya Dhruv  | [Watch Video](#) |

