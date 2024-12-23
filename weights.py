def checkOrder(metrics):
    metric_names = [
        "SP500Momentum",
        "StockPriceBreadth",
        "StockPriceStrength",
        "JunkBondSpread",
        "SafeHavenDemand",
        "PutCallRatio",
        "InsiderTransactions",
        "AAIISentiment",
        "ConsumerSentiment",
        "MarginStats",
        "VIX",
        "T10YearYield",
        "YieldCurve"
    ]
    for i in range(len(metrics)):
        if metrics[i].name != metric_names[i]:
            raise ValueError("The order of the metrics is incorrect.")
    return True

def getWeightsEqual(metrics):
    checkOrder(metrics)
    return [1 for _ in range(len(metrics))]

def getWeightsExYieldCurve(metrics):
    checkOrder(metrics)
    return [1 for _ in range(len(metrics) - 1)] + [0]