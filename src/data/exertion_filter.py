import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class ExertionFilter:
    """
    Isolates cognitive stress by regressing out purely physical exertion
    components (ACC magnitude) from physiological streams.
    """
    def __init__(self):
        self.model = LinearRegression()

    def get_acc_magnitude(self, df):
        """Calculates L2 norm of ACC axes."""
        # Handles various column naming conventions
        cols = [c for c in df.columns if 'ACC' in c.upper()]
        if len(cols) < 3: return np.zeros(len(df))
        return np.sqrt((df[cols]**2).sum(axis=1))

    def remove_physical_component(self, target_signal, acc_mag):
        """Residual-based filtering."""
        X = acc_mag.reshape(-1, 1)
        y = target_signal
        
        # Fit on current context
        self.model.fit(X, y)
        predicted = self.model.predict(X)
        
        # The residual is the target signal minus the physical prediction
        residual = y - predicted
        return residual - np.mean(residual)

    def process(self, df, targets=['EDA', 'HR']):
        """Applies regression to target columns."""
        acc_mag = self.get_acc_magnitude(df)
        if np.all(acc_mag == 0): return df
            
        filtered = df.copy()
        for col in targets:
            if col in df.columns:
                filtered[col] = self.remove_physical_component(df[col].values, acc_mag.values)
        return filtered
