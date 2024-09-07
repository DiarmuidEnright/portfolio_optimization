import pandas as pd

def compare_portfolios(weights_mvo, weights_rp, weights_hrp):
    weights_df = pd.DataFrame({
        'MVO': weights_mvo,
        'Risk Parity': weights_rp,
        'HRP': weights_hrp
    })
    return weights_df