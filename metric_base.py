import datetime as dt

class Metric:
    data_dir = None
    start_date = None
    end_date = None

    def __init__(self):
        self.data = None
        self.processed = None
        self.result = None
        self.name = self.__class__.__name__
    
    @classmethod
    def setPreferences(cls, data_dir, start_date, end_date=dt.date.today()):
        cls.data_dir = data_dir
        cls.start_date = start_date
        cls.end_date = end_date

    def get(self):
        if self.data_dir is None or self.start_date is None or self.end_date is None:
            raise ValueError("Please set the preferences.")
        self.fetch()
        self.calculate()
        self.normalize()
        return self.result
    
    def fetch(self):
        raise NotImplementedError
    
    def calculate(self):
        raise NotImplementedError

    def normalize(self):
        raise NotImplementedError