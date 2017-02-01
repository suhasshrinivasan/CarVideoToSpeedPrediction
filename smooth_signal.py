import numpy as np
import pandas as pd

def ma_smoothing(y_pred, smoothing_window_size, min_periods=1):
    """
    Update signal values with average values of their nearest neighbors
    in both directions and returns the results.
    """
    y_pred_smoothed = pd.rolling_mean(
        y_pred, smoothing_window_size, min_periods=min_periods, center=True)
    missing_idxs = np.isnan(y_pred_smoothed)
    y_pred_smoothed[missing_idxs] = y_pred[missing_idxs]
    return y_pred_smoothed


def ewma_smoothing(y_pred):
    """
    Update signal values with exponentially weighted moving average values
    of their nearest neights in both directions and returns the results

    Credit: http://connor-johnson.com/2014/02/01/smoothing-with-exponentially-weighted-moving-averages/
    """
    ewma = pd.stats.moments.ewma
    fwd = ewma(y_pred, span=15)  # Take EWMA in fwd direction
    bwd = ewma(y_pred[::-1], span=15)  # Take EWMA in bwd direction
    smoothed = np.vstack((fwd, bwd[::-1]))  # Lump fwd and bwd together
    smoothed = np.mean(smoothed, axis=0)  # Average
    return smoothed
