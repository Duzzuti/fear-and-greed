import os
import pandas as pd
import matplotlib.pyplot as plt

import utils
import weights
import basic_utils

data_dir = "data/"
sp500_dir = data_dir + "sp500/"
start_date = "2000-01-01"
fear_and_greed_dir = "fearAndGreedData/"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(sp500_dir):
    os.makedirs(sp500_dir)

if not os.path.exists(fear_and_greed_dir):
    os.makedirs(fear_and_greed_dir)

# Combine normalized metrics
def calculate_fear_greed_index(normalized_metrics, metric_weights):
    normalized_metrics = list(map(lambda x: x.result, normalized_metrics))
    for weight, metric in zip(metric_weights, normalized_metrics):
        metric *= weight
    metrics_weights_tuples = tuple(zip(normalized_metrics, metric_weights))
    metrics_weights_tuples = sorted(metrics_weights_tuples, key=lambda x: x[0].index[0])
    normalized_metrics, sorted_weights = zip(*metrics_weights_tuples)
    current_date = normalized_metrics[0].index[0]
    df = None
    for i in range(1, len(normalized_metrics)):
        if normalized_metrics[i].index[0] != current_date:
            # first calculate df for the current date
            new_df = pd.concat(normalized_metrics[:i], axis=1).sum(axis=1) / sum(sorted_weights[:i])
            if not isinstance(df, pd.Series):
                df = new_df[new_df.index < normalized_metrics[i].index[0]]
            else:
                new_df = new_df[(new_df.index < normalized_metrics[i].index[0]) & (new_df.index > df.index[-1])]
                df = pd.concat([df, new_df])
            current_date = normalized_metrics[i].index[0]
    new_df = pd.concat(normalized_metrics, axis=1).sum(axis=1) / sum(metric_weights)
    new_df = new_df[new_df.index > df.index[-1]]
    df = pd.concat([df, new_df])
    df.index = pd.to_datetime(df.index).date
    return df


metrics = utils.fetch_all(data_dir, start_date, skip_scraping=True)

fear_greed_index = calculate_fear_greed_index(metrics, metric_weights=weights.getWeightsExYieldCurve(metrics))
fear_greed_index = basic_utils.normalize_tanh(fear_greed_index, shift=-50, steepness=0.04)
fear_greed_index.to_csv(fear_and_greed_dir + "ExYieldCurve.csv")

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
