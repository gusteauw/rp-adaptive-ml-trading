import pandas as pd
import numpy as np

class MarketData:
    """
    A class to handle loading and basic preprocessing of market data
    from CSV files or other sources.
    """

    def __init__(self, file_path=None):
        """
        :param file_path: Path to local CSV or None if data will be loaded differently.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load data from CSV (placeholder for now)."""
        if not self.file_path:
            raise ValueError("No file path specified.")
        self.data = pd.read_csv(self.file_path, parse_dates=["date"], index_col="date")
        return self.data

    def preprocess_data(self):
        """Example preprocessing: drop duplicates, handle missing values, add log returns."""
        if self.data is None:
            raise RuntimeError("Data must be loaded before preprocessing.")
        
        # 1. Sort by date (if not already sorted)
        self.data.sort_index(inplace=True)

        # 2. Drop duplicates (just in case)
        self.data.drop_duplicates(inplace=True)

        # 3. Handle missing values (simple forward fill, for instance)
        self.data.ffill(inplace=True)
        self.data.dropna(inplace=True)

        # 4. Create a 'log_returns' column
        if 'close' not in self.data.columns:
            raise KeyError("'close' price column not found in data.")
        self.data['log_returns'] = np.log(self.data['close'] / self.data['close'].shift(1))
        self.data.dropna(inplace=True)

        return self.data
