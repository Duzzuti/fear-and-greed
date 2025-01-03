import pandas as pd
import numpy as np
import os

class Metric:
    data_dir = None
    start_date = None
    end_date = None
    trading_days = None

    def __init__(self, shift=pd.DateOffset(days=0)):
        self.data = None
        self.processed = None
        self.result = None
        self.test = None    # DEBUG
        self.shift = shift
        self.name = self.__class__.__name__
    
    @classmethod
    def setPreferences(cls, data_dir, start_date, end_date=pd.Timestamp.today().date()):
        cls.data_dir = data_dir
        cls.start_date = start_date
        cls.end_date = end_date
        cls.trading_days = pd.read_csv(data_dir + "trading_days.csv", index_col=0, header=[0,1])

    def get(self):
        if self.data_dir is None or self.start_date is None:
            raise ValueError("Please set the preferences.")
        self.fetch()
        self.calculate()
        self.normalize()
        self.reindex()
        return self.result
    
    def fetch(self):
        raise NotImplementedError
    
    def calculate(self):
        raise NotImplementedError

    def normalize(self):
        raise NotImplementedError

    def reindex(self):
        # shift the data
        self.result.index = self.result.index + self.shift
        self.result.index = pd.to_datetime(self.result.index).date
        # add all dates to result
        self.result = self.result.reindex(pd.date_range(self.start_date, self.end_date), method='ffill')
        self.result = self.result.reindex(self.trading_days.index, fill_value=np.nan)
        self.result.dropna(inplace=True)
    
    def save(self):
        if not os.path.exists("indicatorData/"):
            os.mkdir("indicatorData/")
        self.result.to_csv("indicatorData/{}.csv".format(self.name))