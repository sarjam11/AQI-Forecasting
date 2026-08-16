
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")


class NaivePersistence:
   
    name = "Naive Persistence"
    
    def fit(self, train, target="pm25"):
        self.target = target
        return self
    
    def predict(self, history, horizon=24):
        last_val = history[self.target].iloc[-1]
        return np.full(horizon, last_val)
    
    def predict_all(self, df, horizon=24):
        """Generate predictions for every row in df."""
        target = self.target
        preds = df[target].shift(1).values  # just the previous value
        return preds


class SeasonalNaive:
  
    name = "Seasonal Naive (24h)"
    
    def __init__(self, season_length=24):
        self.season_length = season_length
    
    def fit(self, train, target="pm25"):
        self.target = target
        return self
    
    def predict(self, history, horizon=24):
        season = self.season_length
        last_cycle = history[self.target].iloc[-season:].values
        reps = (horizon // season) + 1
        return np.tile(last_cycle, reps)[:horizon]
    
    def predict_all(self, df, horizon=24):
        return df[self.target].shift(self.season_length).values   # value from 24 hours ago


class MovingAverage:
    
    name = "Moving Average"
    
    def __init__(self, window=24):
        self.window = window
    
    def fit(self, train, target="pm25"):
        self.target = target
        self.name = f"Moving Average ({self.window}h)"
        return self
    
    def predict(self, history, horizon=24):
        avg = history[self.target].iloc[-self.window:].mean()
        return np.full(horizon, avg)
    
    def predict_all(self, df, horizon=24):
        return df[self.target].rolling(window=self.window, min_periods=1).mean().values  # rolling mean


class LinearBaseline:
    
    name = "Linear (Ridge)"
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
    
    def fit(self, train, target="pm25"):
        self.target = target
        feature_cols = [c for c in train.columns if c != target]
        self.feature_cols = feature_cols
        
        X = train[feature_cols].values
        y = train[target].values
        
        # Handle any remaining NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        self.model.fit(X[mask], y[mask])
        
        # Store feature importances
        self.feature_importance = pd.Series(
            np.abs(self.model.coef_),
            index=feature_cols
        ).sort_values(ascending=False)
        
        return self
    
    def predict_all(self, df, horizon=24):
        X = df[self.feature_cols].values
        return self.model.predict(X)
    
    def get_top_features(self, n=20):
        return self.feature_importance.head(n)


class SARIMABaseline:

    name = "SARIMA"
    
    def __init__(self, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
    
    def fit(self, train, target="pm25"):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        self.target = target
        
        # Aggregate to daily
        daily = train[target].resample("1D").mean().dropna()
        
        print(f"  Fitting SARIMA{self.order}x{self.seasonal_order}...")
        print(f"  Training on {len(daily)} daily observations...")
        
        model = SARIMAX(
            daily,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        
        self.results = model.fit(disp=False, maxiter=300)
        self.daily_train = daily
        
        print(f"  AIC: {self.results.aic:.1f}")
        
        return self
    
    def predict_daily(self, n_days):
        """Forecast n_days ahead (daily resolution)."""
        forecast = self.results.get_forecast(steps=n_days)
        return forecast.predicted_mean, forecast.conf_int()
    
    def predict_all_daily(self, test_daily_index):
        """Generate daily predictions for the test period."""
        n_days = len(test_daily_index)
        pred_mean, pred_ci = self.predict_daily(n_days)
        pred_mean.index = test_daily_index
        pred_ci.index = test_daily_index
        return pred_mean, pred_ci
    
    def auto_tune(self, train, target="pm25", max_p=3, max_q=3):
        
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import itertools
        
        self.target = target
        daily = train[target].resample("1D").mean().dropna()
        
        best_aic = np.inf
        best_order = self.order
        
        p_range = range(0, max_p + 1)
        q_range = range(0, max_q + 1)
        d_range = [1]  # usually 1 for AQI
        
        total = (max_p + 1) * (max_q + 1)
        print(f"  Auto-tuning SARIMA: testing {total} combinations...")
        
        for p, d, q in itertools.product(p_range, d_range, q_range):
            try:
                model = SARIMAX(
                    daily,
                    order=(p, d, q),
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                results = model.fit(disp=False, maxiter=200)
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
                    
            except Exception:
                continue
        
        print(f"  Best order: SARIMA{best_order} (AIC={best_aic:.1f})")
        self.order = best_order
        
        # Refit with best order
        self.fit(train, target)
        
        return self
