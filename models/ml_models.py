from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from typing import Tuple

def train_predictive_model(data: pd.DataFrame, target: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor()
    model.fit(data, target)
    return model

def predict_returns(model: RandomForestRegressor, data: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(data))