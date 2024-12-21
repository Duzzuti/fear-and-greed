import datetime as dt
import pandas as pd
import numpy as np
import os

import utils

class Metric:
    data_dir = None
    start_date = None
    end_date = None

    def __init__(self):
        self.data = None
        self.processed = None
        self.result = None
        self.test = None    # DEBUG
        self.name = self.__class__.__name__
    
    @classmethod
    def setPreferences(cls, data_dir, start_date, end_date=dt.date.today()):
        cls.data_dir = data_dir
        cls.start_date = start_date
        cls.end_date = end_date

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
        # get the date_count.csv
        date_count = utils.get_repo_data("date_count.csv", self.start_date, self.end_date)
        # add all dates to result
        self.result = self.result.reindex(pd.date_range(self.start_date, self.end_date), fill_value=np.nan)
        self.result.ffill(inplace=True)
        self.result = self.result.reindex(date_count.index, fill_value=np.nan)
    
    def save(self):
        if not os.path.exists("indicatorData/"):
            os.mkdir("indicatorData/")
        self.result.to_csv("indicatorData/{}.csv".format(self.name))