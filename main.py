import os
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *
from metric_base import Metric
import utils

data_dir = "data/"
sp500_dir = data_dir + "sp500/"
start_date = "2000-01-01"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(sp500_dir):
    os.makedirs(sp500_dir)

# Combine normalized metrics
def calculate_fear_greed_index(normalized_metrics):
    normalized_metrics = list(map(lambda x: x.result, normalized_metrics))
    return pd.concat(normalized_metrics, axis=1).mean(axis=1)


metrics = utils.fetch_all(data_dir, start_date)

fear_greed_index = calculate_fear_greed_index(metrics)


# Plot the Fear and Greed Index
plt.figure(figsize=(12, 6))
#plt.plot(aaii_sentiment.index, aaii_sentiment, label="10Y Bond Yield", color="blue")
#plt.plot(normalized_safe_haven_demand.index, normalized_safe_haven_demand, label="10Y Bond Yield (Normalized)", color="green")
plt.plot(fear_greed_index.index, fear_greed_index, label="Fear & Greed Index", color="purple")
#plt.plot(fear_greed_index2.index, fear_greed_index2, label="Fear & Greed Index (Smoothed)", color="orange")
plt.title("Fear & Greed Index")
plt.xlabel("Date")
plt.ylabel("Index Value")
plt.axhline(50, color='grey', linestyle='--', label="Neutral")
plt.legend()
plt.show()
