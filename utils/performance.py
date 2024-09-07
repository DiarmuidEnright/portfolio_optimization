import pandas as pd
from typing import Dict

def compare_portfolios(weights_mvo: Dict[str, float], weights_rp: Dict[str, float], weights_hrp: Dict[str, float]) -> pd.DataFrame:
    weights_df = pd.DataFrame({
        'MVO': weights_mvo,
        'Risk Parity': weights_rp,
        'HRP': weights_hrp
    })
    return weights_df