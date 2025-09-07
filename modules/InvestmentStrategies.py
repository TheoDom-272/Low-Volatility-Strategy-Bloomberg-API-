import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
from datetime import date as _date, datetime as _datetime

from tqdm import tqdm
tqdm.pandas()




class BetaCalculator:
    """
    Class that encapsulates the different beta calculation methods
    """

    def rescale_returns(self, returns):
        """Rescale returns to avoid convergence issues."""
        scale_factor = 100
        return returns * scale_factor, scale_factor
    

    def realized_beta_vectorized(self,
                                 stock_returns: pd.DataFrame,
                                 market_returns: pd.Series | pd.DataFrame,
                                 window: int = 90) -> pd.DataFrame:
        """
        Vectorized realized beta
        """

        # Strict date alignment
        stock_returns, market_returns = stock_returns.align(
            market_returns, join="inner", axis=0
        )

        stock  = stock_returns.values.astype(float)             
        market = market_returns.squeeze().values.astype(float)  

        n, m = stock.shape
        if n < window:
            raise ValueError("Pas assez d’observations : len(series) < window")

        # Sliding windows
        stock_w  = sliding_window_view(stock,  window, axis=0)   
        market_w = sliding_window_view(market, window, axis=0)   


        if stock_w.shape[1] == m and stock_w.shape[2] == window:
            stock_w = np.swapaxes(stock_w, 1, 2)                

        # Means
        stock_mean  = np.nanmean(stock_w,  axis=1)              
        market_mean = np.nanmean(market_w, axis=1)               

        # Covariances
        prod_mean   = np.nanmean(stock_w * market_w[:, :, None], axis=1)  
        covariance  = prod_mean - stock_mean * market_mean[:, None]

        # Market variance
        market_var  = np.nanmean(market_w**2, axis=1) - market_mean**2    

        beta = covariance / market_var[:, None]              

        # Result DataFrame
        beta_dates = stock_returns.index[window - 1:]
        return pd.DataFrame(beta, index=beta_dates,
                            columns=stock_returns.columns)



    def calculate_realized_beta(self, stock_returns_df, market_returns_df, stock_name, window=90):
        """
        Calculate the realized beta using a rolling window.
        """

        if len(stock_returns_df) != len(market_returns_df):
            raise ValueError("Return series must have the same length.")

        rolling_cov = stock_returns_df.rolling(window).cov(market_returns_df)
        rolling_var = market_returns_df.rolling(window).var()

        realized_beta = rolling_cov / rolling_var
        realized_beta = realized_beta.dropna()

        return pd.Series(realized_beta.iloc[:, 0], index=realized_beta.index)




    """As you can see when running the "Etudes" script, we were not able to implement beta using the DCC approach. The code is left here so our mistakes can be examined, but it is not callable from the user interface."""

    def calculate_dcc_beta(self, stock_returns, market_returns):
        """Compute the dynamic DCC beta."""

        # Ensure both inputs are Series
        if not isinstance(stock_returns, pd.Series):
            stock_returns = stock_returns.iloc[:, 0]
        if not isinstance(market_returns, pd.Series):
            market_returns = market_returns.iloc[:, 0]

        # Align on common dates
        df = pd.concat([stock_returns, market_returns], axis=1, join="inner")
        df.columns = ["stock", "market"]
        df = df.dropna()
        if df.empty:
            return pd.Series(dtype=float)

        stock_clean = df["stock"]
        market_clean = df["market"]

        # Rescale returns (×100) to avoid near-zero magnitudes
        stock_rs, stock_scale = self.rescale_returns(stock_clean)
        market_rs, market_scale = self.rescale_returns(market_clean)

        # Select GARCH orders via the AIC criterion
        stock_order, stock_model = self.select_best_garch_order(stock_rs)
        market_order, market_model = self.select_best_garch_order(market_rs)

        # Retrieve the conditional volatility series
        stock_volatility = stock_model.conditional_volatility
        market_volatility = market_model.conditional_volatility

        # Standardize residuals
        stock_residuals = stock_model.resid / stock_volatility
        market_residuals = market_model.resid / market_volatility

        # Stack residuals into one array
        residuals = np.stack([stock_residuals.values, market_residuals.values], axis=1)

        # Estimate DCC parameters by maximizing the log-likelihood
        a, b = self.estimate_dcc_params(residuals)

        # Compute time-varying betas
        Q_bar = np.cov(residuals.T)  # Covariance matrix of residuals
        T = len(stock_residuals)  # Number of time steps
        Q_t = Q_bar.copy()  # Copy of initial covariance to keep Q̄ intact
        beta_t = []  # Initialize list of betas

        for t in range(T):
            # Extract the residual vector for time t
            z_t = np.array([stock_residuals.iloc[t], market_residuals.iloc[t]])[:, None]
            # Update the dynamic covariance matrix (DCC recursion)
            Q_t = (1 - a - b) * Q_bar + a * (z_t @ z_t.T) + b * Q_t
            # Regularization: add small epsilon to avoid division by zero
            diag_Q = np.sqrt(np.diag(Q_t) + 1e-8)
            # Derive the correlation matrix by normalizing Q_t
            R_t = Q_t / np.outer(diag_Q, diag_Q)

            # Conditional covariance between stock and market
            cov_t = R_t[0, 1] * stock_volatility.iloc[t] * market_volatility.iloc[t]
            # Conditional variance of the market
            var_m_t = market_volatility.iloc[t] ** 2
            # Dynamic beta
            beta_t.append(cov_t / var_m_t)

        # Dynamic beta for each timestamp
        return pd.Series(beta_t, index=stock_clean.index) / (stock_scale / market_scale)

    def select_best_garch_order(self, returns, max_p=3, max_q=3):
        """Select the best GARCH(p, q) orders using the AIC criterion."""
        best_aic = np.inf
        best_order = None
        best_model = None

        if not isinstance(returns, pd.Series):
            returns_series = returns.iloc[:, 0]
        else:
            returns_series = returns

        returns_series.index = pd.to_datetime(returns_series.index)
        returns_series = returns_series.sort_index()

        # Drop NaNs and cast to float
        returns_series = returns_series.dropna().astype(float)

        # Estimate a GARCH model for every lag combination p and q
        for p in range(1, max_p + 1):
            for q in range(1, max_q + 1):
                try:
                    model = arch_model(returns_series, vol='Garch', p=p, q=q)
                    result = model.fit(disp='off')
                    # Keep only the model with the lowest AIC
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, q)
                        best_model = result
                except Exception:
                    continue

        return best_order, best_model

    def estimate_dcc_params(self, residuals):
        """
        Compute the (a, b) parameters of a DCC(1, 1) model under the
        constraint a + b < 1.
        """

        # Empirical unconditional correlation matrix Q̄
        Q_bar = np.cov(residuals.T)
        d = np.sqrt(np.diag(Q_bar))
        Q_bar = Q_bar / np.outer(d, d)

        def neg_loglik(params):
            a, b = params
            T = residuals.shape[0]
            Q_t = Q_bar.copy()
            ll = 0.0
            #
            for t in range(T):
                z = residuals[t][:, None]
                Q_t = (1 - a - b) * Q_bar + a * (z @ z.T) + b * Q_t
                # Normalize to obtain a proper correlation matrix
                d_t = np.sqrt(np.diag(Q_t))
                R_t = Q_t / np.outer(d_t, d_t)
                # Guard against non-positive determinants
                det = np.linalg.det(R_t)
                if det <= 0:
                    return 1e8
                ll += np.log(det) + float(z.T @ np.linalg.inv(R_t) @ z)
            return ll

        cons = ({'type': 'ineq', 'fun': lambda x: 1 - x[0] - x[1]})
        bounds = [(0, 1), (0, 1)]
        res = minimize(neg_loglik, [0.01, 0.98], bounds=bounds, constraints=cons)
        return res.x  # [a, b]

    def beta_shrinkage_in_chunks(self,stock_returns, market_returns,
                                 short_window=90, annual_window=252,
                                 prior_window=252, n_chunks=4):
        """
        Calcule le beta shrinkage par morceaux pour réduire l'empreinte mémoire.
        """
        calculator = BetaCalculator()
        chunks = np.array_split(stock_returns.columns, n_chunks)
        beta_frames = []

        for cols in chunks:
            sub_stock = stock_returns[cols]
            beta_sub = calculator.beta_shrinkage_vectorized(
                sub_stock,
                market_returns,
                short_window=short_window,
                annual_window=annual_window,
                prior_window=prior_window
            )
            beta_frames.append(beta_sub)

        # Recompose le DataFrame complet
        beta_all = pd.concat(beta_frames, axis=1)
        return beta_all


    def beta_shrinkage_vectorized(
            self,
            stock_returns,
            market_returns,
            short_window: int = 90,
            annual_window: int = 252,
            prior_window: int = 252,
    ):
        """
        Vectorized computation of shrunk beta for multiple assets.

        """

        # Retrieve metadata if `stock_returns` is a DataFrame
        if isinstance(stock_returns, pd.DataFrame):
            dates_all = stock_returns.index
            columns = stock_returns.columns
        else:
            dates_all = None
            columns = None

        if isinstance(stock_returns, pd.DataFrame) and isinstance(market_returns, pd.DataFrame):
            stock_returns, market_returns = stock_returns.align(market_returns, join='inner', axis=0)
        else:
            # Si ce sont des Series, même logique :
            stock_returns = pd.Series(stock_returns).dropna()
            market_returns = pd.Series(market_returns).dropna()
            stock_returns, market_returns = stock_returns.align(market_returns, join='inner')

        # Maintenant vous pouvez extraire les valeurs sûres
        stock = stock_returns.values  # (n, m)
        market = market_returns.values.squeeze()  # (n,)

        # Convert to numpy array if necessary
        # stock = np.asarray(stock_returns)  # shape (n, m)
        # market = np.asarray(market_returns).squeeze()  # shape (n,)
        n, m = stock.shape


        # 1. Compute short-window beta and its variance (V)
        T_short = n - short_window + 1
        stock_windows_short = (
            sliding_window_view(stock, short_window, axis=0).transpose(0, 2, 1)
        )  # (T_short, short_window, m)
        market_windows_short = sliding_window_view(
            market, short_window, axis=0
        )  # (T_short, short_window)

        # Means for each window
        market_mean_short = np.nanmean(market_windows_short, axis=1)  # (T_short,)
        stock_mean_short = np.nanmean(stock_windows_short, axis=1)  # (T_short, m)

        # Sum of squares for the market for each window
        S_x = np.sum((market_windows_short - market_mean_short[:, None]) ** 2, axis=1)  # (T_short,)

        diff_market = (market_windows_short - market_mean_short[:, None])[:, :, None]
        diff_stock = stock_windows_short - stock_mean_short[:, None]  # (T_short, short_window, m)

        # Covariance between market and each asset over the window
        cov = np.sum(diff_market * diff_stock, axis=1)  # (T_short, m)

        beta_short = cov / S_x[:, None]  # (T_short, m)

        # Intercept
        intercept = stock_mean_short - beta_short * market_mean_short[:, None]  # (T_short, m)

        # Residuals
        residuals = stock_windows_short - (
                intercept[:, None, :] + beta_short[:, None, :] * market_windows_short[:, :, None]
        )

        # Estimate sigma over the short window (with n−2 degrees of freedom)
        sigma2 = np.sum(residuals ** 2, axis=1) / (short_window - 2)  # (T_short, m)

        # Variance of the beta estimator
        V = sigma2 / S_x[:, None]  # (T_short, m)


        # 2. Compute long-window beta (annual window)
        T_long = n - annual_window + 1
        stock_windows_long = (
            sliding_window_view(stock, annual_window, axis=0).transpose(0, 2, 1)
        )  # (T_long, annual_window, m)
        market_windows_long = sliding_window_view(
            market, annual_window, axis=0
        )  # (T_long, annual_window)

        market_mean_long = np.nanmean(market_windows_long, axis=1)  # (T_long,)
        stock_mean_long = np.nanmean(stock_windows_long, axis=1)  # (T_long, m)

        S_x_long = np.sum((market_windows_long - market_mean_long[:, None]) ** 2, axis=1)  # (T_long,)

        diff_market_long = (market_windows_long - market_mean_long[:, None])[:, :, None]
        diff_stock_long = stock_windows_long - stock_mean_long[:, None]

        cov_long = np.sum(diff_market_long * diff_stock_long, axis=1)  # (T_long, m)
        beta_long = cov_long / S_x_long[:, None]  # (T_long, m)


        # 3. Time alignment
        # beta_short and V start at index short_window−1,
        # beta_long starts at index annual_window−1.
        # Choose the common starting point: t_start = max(short_window, annual_window) − 1.
        t_start = max(short_window, annual_window) - 1
        # For beta_short (and V): indices starting at t_start − (short_window − 1)
        beta_short_common = beta_short[(t_start - short_window + 1):, :]  # (n − t_start, m)
        V_common = V[(t_start - short_window + 1):, :]  # same shape
        # For beta_long: indices starting at t_start − (annual_window − 1)
        beta_long_common = beta_long[(t_start - annual_window + 1):, :]  # (n − t_start, m)

        common_length = beta_short_common.shape[0]  # identical for beta_long_common and V_common


        # 4. Compute tau2 : variance of the short betas over a prior_window horizon
        if common_length < prior_window:
            raise ValueError(
                "Not enough observations to compute tau2 with the given prior_window."
            )

        # Build rolling windows on beta_short_common along the time axis
        beta_short_prior = sliding_window_view(beta_short_common, prior_window, axis=0).transpose(
            0, 2, 1
        )

        T_tau = beta_short_prior.shape[0]
        tau2 = np.nanvar(beta_short_prior, axis=1, ddof=1)  # (T_tau, m)

        # Keep the last T_tau observations of beta_short_common, V_common and beta_long_common
        beta_short_final = beta_short_common[-T_tau:, :]  # (T_tau, m)
        V_final = V_common[-T_tau:, :]  # (T_tau, m)
        beta_long_final = beta_long_common[-T_tau:, :]  # (T_tau, m)


        # 5. Compute the shrunk beta
        weight_data = tau2 / (tau2 + V_final)
        weight_prior = V_final / (tau2 + V_final)
        beta_shrunk = weight_data * beta_short_final + weight_prior * beta_long_final  # (T_tau, m)

        # Set the date index corresponding to the calculations:
        # The first common date is at index t_start in the original DataFrame
        if dates_all is not None:
            # Extract dates from t_start then keep the last T_tau dates
            common_dates = dates_all[t_start:]
            beta_dates = common_dates[-T_tau:]
        else:
            beta_dates = np.arange(T_tau)

        beta_shrunk_df = pd.DataFrame(beta_shrunk, index=beta_dates, columns=columns)

        return beta_shrunk_df

    # Not used; will be removed eventually
    def calculate_beta_shrinkage(
            self,
            stock_returns,
            market_returns,
            window: int = 90,
            prior_window: int = 252,
            annual_window: int = 252,
    ):
        """
        Compute the shrunk beta following a Bayesian approach for a single asset.
        """
        
        import statsmodels.api as sm
        import pandas as pd
        import numpy as np

        # Convert to Series if necessary
        if not isinstance(stock_returns, pd.Series):
            stock_returns = pd.Series(stock_returns)
        if isinstance(market_returns, pd.DataFrame):
            if market_returns.shape[1] == 1:
                market_returns = market_returns.iloc[:, 0]
            else:
                raise ValueError(
                    "market_returns must be a Series or a single-column DataFrame."
                )

        # Ensure alignment and sorting of dates
        stock_returns = stock_returns.sort_index()
        market_returns = market_returns.sort_index()

        dates = stock_returns.index
        # Start at max(window, annual_window, prior_window) to guarantee enough data
        start_index = max(window, annual_window, prior_window)
        beta_shrinkage_series = pd.Series(index=dates[start_index:], dtype=float)

        # Store beta_short estimates to compute tau² empirically
        beta_short_history = []
        default_tau2 = 0.01  # Default value if not enough estimates yet

        # Loop over each date from start_index onward
        for t in range(start_index, len(dates)):
            # Short window for beta_short
            window_dates = dates[t - window: t]
            # Annual window for beta_annual
            annual_dates = dates[t - annual_window: t]

            # Double-check we indeed have enough observations (start_index should ensure this)
            if len(window_dates) < window or len(annual_dates) < annual_window:
                continue

            # OLS estimation on the short window
            X_short = sm.add_constant(market_returns.loc[window_dates])
            y_short = stock_returns.loc[window_dates]
            model_short = sm.OLS(y_short, X_short)
            results_short = model_short.fit()
            beta_short = results_short.params.iloc[1]  # use .iloc to avoid warnings
            V = results_short.bse.iloc[1] ** 2  # Variance of the estimate

            # Save this estimate to compute tau² later
            beta_short_history.append(beta_short)
            # For tau² we use the last *prior_window* beta_short values (if available)
            if len(beta_short_history) >= prior_window:
                recent_betas = beta_short_history[-prior_window:]
                tau2 = np.var(recent_betas)
            else:
                tau2 = default_tau2

            # OLS estimation on the annual window
            X_annual = sm.add_constant(market_returns.loc[annual_dates])
            y_annual = stock_returns.loc[annual_dates]
            model_annual = sm.OLS(y_annual, X_annual)
            results_annual = model_annual.fit()
            beta_annual = results_annual.params.iloc[1]

            # Compute Bayesian weights
            weight_data = tau2 / (tau2 + V)
            weight_prior = V / (tau2 + V)
            beta_shrunk = weight_data * beta_short + weight_prior * beta_annual

            beta_shrinkage_series.loc[dates[t]] = beta_shrunk

        return beta_shrinkage_series



class PortfolioManagement:

    def __init__(self,prices_df, beta_df, volume_df, returns_df,method_optim,short_allocation = 1,min_stock = 0.15, beta_threshold=0.20, volume_threshold=-1, window=126,market_neutral=False):
        """
        """
        self.price_df = prices_df
        self.beta_df = beta_df
        self.volume_df = volume_df
        self.returns_df = returns_df
        self.beta_threshold = beta_threshold
        self.volume_threshold = volume_threshold
        self.window = window
        self.min_stock = min_stock
        self.selected_stocks_long = None
        self.selected_stocks_short = None #Séléction des stock à la date
        self.method_optim = method_optim
        self.short_allocation = short_allocation
        self.market_neutral = market_neutral

    def _align_date(self, df: pd.DataFrame, dt):

        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce")

        # Make a tz-naive copy for uniform comparisons
        idx = df.index.tz_convert(None) if df.index.tz is not None else df.index

        # Coerce dt to tz-naive Timestamp
        if not isinstance(dt, pd.Timestamp):
            if isinstance(dt, (_date, _datetime)):
                dt = pd.Timestamp(dt)
            else:                         # string, etc.
                dt = pd.to_datetime(dt, errors="raise")
        dt = dt.tz_convert(None) if dt.tzinfo else dt

        # Locate insertion point: last label ≤ dt
        pos = idx.searchsorted(dt, side="right") - 1
        if pos < 0:
            raise ValueError(
                f"{dt.date()} is earlier than the first available date "
                f"({idx[0].date()}) in the DataFrame."
            )

        return idx[pos] 

    def cross_sectional_liquidity_filter_with_history(self,
            all_prices_df: pd.DataFrame,
            all_volumes_df: pd.DataFrame,
            date,
            window_days: int,
            negative_threshold: float = -1.96,
            positive_threshold: float = 1.96,
            compare_log: bool = False,  # Nouveau paramètre
    ):
        """
        Liquidity filter with optional comparison of raw vs log indicators.
        """

        # Validation
        if date not in all_prices_df.index:
            raise ValueError(f"Date {date} is not present in the price DataFrame.")
        if date not in all_volumes_df.index:
            raise ValueError(f"Date {date} is not present in the volume DataFrame.")

        # Rolling means up to date
        rolling_prices = all_prices_df.loc[:date].tail(window_days).mean()
        rolling_volumes = all_volumes_df.loc[:date].tail(window_days).mean()

        # Raw dollar-volume indicator
        liquidity_indicator = rolling_prices * rolling_volumes

        # Log-transformed indicator
        log_indicator = np.log1p(liquidity_indicator)

        # Optionnel : tracer les distributions pour comparaison
        if compare_log:
            # 1) Distribution brute
            mean_raw = liquidity_indicator.mean()
            std_raw = liquidity_indicator.std()
            upper_raw = mean_raw + positive_threshold * std_raw
            lower_raw = mean_raw + negative_threshold * std_raw

            plt.figure()
            plt.hist(liquidity_indicator, bins=50)
            plt.axvline(mean_raw, linestyle='--')
            plt.axvline(upper_raw, linestyle='--')
            plt.axvline(lower_raw, linestyle='--')
            plt.title("Distribution brute (prix × volume)\navec mean ±1.96·std")
            plt.xlabel("Dollar-volume brut")
            plt.ylabel("Effectif")
            plt.tight_layout()
            plt.show()

            # 2) Distribution log + median/MAD
            med = log_indicator.median()
            mad = (log_indicator - med).abs().median()
            mad = mad if mad > 0 else 1e-8
            upper_log = med + positive_threshold * mad
            lower_log = med + negative_threshold * mad

            plt.figure()
            plt.hist(log_indicator, bins=50)
            plt.axvline(med, linestyle='--')
            plt.axvline(upper_log, linestyle='--')
            plt.axvline(lower_log, linestyle='--')
            plt.title("Distribution log1p\navec médiane ±1.96·MAD")
            plt.xlabel("log1p(dollar-volume)")
            plt.ylabel("Effectif")
            plt.tight_layout()
            plt.show()

        # Robust z-score helper
        def robust_z(series: pd.Series):
            med = series.median()
            mad = (series - med).abs().median()
            mad = mad if mad > 0 else 1e-8
            return (series - med) / mad

        # Phase 1 on log-indicator
        z_full = robust_z(log_indicator)
        good_assets = z_full[z_full > positive_threshold]

        # Phase 2 on remainder
        remainder = log_indicator.drop(good_assets.index)
        if not remainder.empty:
            z_rem = robust_z(remainder)
        else:
            z_rem = pd.Series(dtype=float)

        filtered_rem = z_rem[z_rem >= negative_threshold]
        final_assets = pd.concat([good_assets, filtered_rem])
        excluded_assets = z_rem[z_rem < negative_threshold]

        return final_assets, excluded_assets


    def select_stocks(self, date=None, window: int = 63):
        # If no date is provided, use the last date available in the DataFrame
        if date is None:
            date = self.beta_df.index[-1]

        date = pd.to_datetime(date)
        date_ret  = self._align_date(self.returns_df, date)
        date_beta = self._align_date(self.beta_df, date_ret)
        date_vol = self._align_date(self.volume_df, date_ret)


        print(f"Selection date used: {date}")

        # Extract betas, prices and volumes for the specified date
        try:
            betas_at_date = self.beta_df.loc[date_beta]
            range_beta = self.beta_df.loc[:date_beta].tail(window)
            range_price = self.price_df.loc[:date_ret].tail(window)
            volumes_at_date = self.volume_df.loc[date_vol]
            range_volume = self.volume_df.loc[:date_vol].tail(window)
        except KeyError:
            raise ValueError(f"The date {date} is not present in the DataFrames.")

        # Liquidity filter on assets
        filtered_assets, excluded_assets = self.cross_sectional_liquidity_filter_with_history(
            self.price_df,
            self.volume_df,
            date,
            window,
            negative_threshold=self.volume_threshold,
        )

        print(f"Number of assets passing the liquidity filter: {len(filtered_assets)}")
        print(f"Number of assets excluded by the liquidity filter: {len(excluded_assets)}")

        # Compute the mean beta over the chosen window
        beta_moy = range_beta.mean(axis=0)

        # Threshold initialization
        beta_threshold = self.beta_threshold  # Initial threshold (e.g., 0.25)
        min_stocks = self.min_stock * len(betas_at_date)  # Minimum number of stocks required
        print(f"Minimum number of stocks required: {min_stocks}")

        # Loop until we have enough securities
        while True:
            # Ensure beta_threshold remains between 0 and 1
            if not (0 <= beta_threshold <= 1):
                raise ValueError(f"beta_threshold must be between 0 and 1, current value: {beta_threshold}")

            # Compute the quantile used to pick low betas
            beta_long_quantile = beta_moy.quantile(beta_threshold)
            print(f"Low-beta quantile at {beta_threshold}: {beta_long_quantile}")

            # Select securities with the lowest betas (based on current threshold and rolling-window mean)
            selected_by_beta_for_long = beta_moy[beta_moy < beta_long_quantile].index
            print(f"Number of securities selected by beta for the long bucket: {len(selected_by_beta_for_long)}")

            # Intersect low-beta names with those passing the liquidity filter
            selected_stocks_long = list(set(selected_by_beta_for_long).intersection(filtered_assets.index))
            print(f"Number of long candidates after liquidity intersection: {len(selected_stocks_long)}")

            if self.short_allocation != 0:
                # Compute the quantile used to pick high betas
                beta_short_quantile = beta_moy.quantile(1 - beta_threshold)
                print(f"High-beta quantile at {1 - beta_threshold}: {beta_short_quantile}")

                # Select securities with the highest betas (same rolling-window mean)
                selected_by_beta_for_short = beta_moy[beta_moy > beta_short_quantile].index
                print(f"Number of securities selected by beta for the short bucket: {len(selected_by_beta_for_short)}")

                # Intersect high-beta names with the liquidity filter
                selected_stocks_short = list(set(selected_by_beta_for_short).intersection(filtered_assets.index))
                print(f"Number of short candidates after liquidity intersection: {len(selected_stocks_short)}")
            else:
                selected_stocks_short = "Nothing"

            # Exit loop once we have enough long names
            if len(selected_stocks_long) >= min_stocks:
                break

            # Otherwise, widen the beta threshold
            beta_threshold += 0.025  # Increment threshold by 2.5 pp

            # If the threshold exceeds 100 % we keep going (rare in practice)
            if beta_threshold > 1.0:
                print("Beta threshold exceeds 100 %, continuing search for more names.")

        print(f"Final beta threshold used: {beta_threshold}")

        return selected_stocks_long, selected_stocks_short

    def sharpe_ratio(self, portfolio_returns, eps: float = 1e-8):
        """Compute the Sharpe ratio (mean / standard deviation) of a return series."""
        mean_ret = portfolio_returns.mean()
        std_ret = portfolio_returns.std()
        return mean_ret / (std_ret + eps)

    def optimize_portfolio_mv(
            self,
            selected_stocks_long,
            selected_stocks_short,
            actual_weights=None,
            date=None,
    ):
        """
        Optimise the weight allocation for the LONG and SHORT sleeves
        using a mean-/variance-ratio criterion.

        LONG sleeve : maximise the mean/variance ratio
        SHORT sleeve : short the assets with the worst ratio.

        The optimised weights are then rescaled so that the long pocket
        represents 80 % of total exposure and the short pocket 20 %.

        """
        # If no date is specified, use the most recent date available
        if date is None:
            date = self.returns_df.index[-1]

        if self.market_neutral:
            long_target = self.short_allocation
            short_target = self.short_allocation  
        else:
            long_target = 1.0 + self.short_allocation
            short_target = self.short_allocation 

        # LONG OPTIMISATION
        returns_window_long = self.returns_df[selected_stocks_long].loc[:date].tail(self.window)

        # Daily portfolio return = weighted sum of asset returns
        def portfolio_return_long(w):
            return returns_window_long.dot(w)

        # Objective: maximise Sharpe (mean/std) → minimise the negative Sharpe
        def objective_long(w):
            port_ret = portfolio_return_long(w)
            return -self.sharpe_ratio(port_ret)

        # Equality constraint: sum of weights = long_target
        cons_long = [{'type': 'eq', 'fun': lambda w: np.sum(w) - long_target}]

        # Turnover constraint if current weights are supplied
        if actual_weights is not None:
            current_long = actual_weights.reindex(selected_stocks_long).fillna(0).values
            if np.sum(current_long) != 0:  # Check to avoid division by zero
                current_long = current_long * (long_target / np.sum(current_long))  # Re-scale
            else:
                current_long = np.ones(len(selected_stocks_long)) / len(selected_stocks_long) * long_target
            init_long = current_long
        else:
            init_long = np.ones(len(selected_stocks_long)) / len(selected_stocks_long) * long_target

        bounds_long = [(0, long_target) for _ in selected_stocks_long]

        res_long = minimize(
            objective_long,
            init_long,
            method='SLSQP',
            bounds=bounds_long,
            constraints=cons_long,
            options={'ftol': 1e-9, 'maxiter': 1000},
        )

        if not res_long.success:
            raise ValueError("LONG optimisation failed: " + res_long.message)

        w_long_opt = res_long.x  # Optimal long weights
        portfolio_long = pd.DataFrame(w_long_opt, index=selected_stocks_long, columns=["Weight"])

        # SHORT OPTIMISATION
        # Skip if no short sleeve is required
        if self.short_allocation != 0:
            returns_window_short = self.returns_df[selected_stocks_short].loc[:date].tail(self.window)

            def portfolio_return_short(w):
                return returns_window_short.dot(w)

            # For shorts, weights are negative and must sum to -short_target
            def objective_short(w):
                port_ret = portfolio_return_short(w)
                return -self.sharpe_ratio(port_ret)

            cons_short = [{'type': 'eq', 'fun': lambda w: np.sum(w) + short_target}]
            init_short = -np.ones(len(selected_stocks_short)) / len(selected_stocks_short) * short_target
            bounds_short = [(-1, 0) for _ in selected_stocks_short]

            res_short = minimize(
                objective_short,
                init_short,
                method='SLSQP',
                bounds=bounds_short,
                constraints=cons_short,
                options={'ftol': 1e-9, 'maxiter': 1000},
            )

            if not res_short.success:
                raise ValueError("SHORT optimisation failed: " + res_short.message)

            w_short_opt = res_short.x  # Optimal short weights
            portfolio_short = pd.DataFrame(w_short_opt, index=selected_stocks_short, columns=["Weight"])
        else:
            portfolio_short = pd.DataFrame()  # Empty DataFrame if no short sleeve

        # COMBINE LONG & SHORT SLEEVES
        portfolio_weights = pd.concat([portfolio_long, portfolio_short])

        return portfolio_weights

    def optimize_portfolio_erc(
            self,
            selected_stocks_long,
            selected_stocks_short,
            window: int = 62,
            actual_weights=None,
            date=None,
    ):
        """
        Optimise the weight allocation of the LONG and SHORT sleeves
        using an Equal-Risk-Contribution (ERC) approach.

        LONG sleeve
        -----------
        Find positive weights *w* such that every asset contributes equally
        to portfolio risk.  For the risk-contribution vector
        ``RC = w * (Σw)``, minimise the dispersion across elements of *RC*.

        SHORT sleeve
        ------------
        Apply the same optimisation on positive “intensity” weights, then
        multiply the result by −1 so the final weights are negative, with the
        additional constraint that the sum of absolute weights equals
        ``short_allocation``.

        All returns used for the optimisation are taken over a *window*
        of observations ending at *date*.
        """

        # Default to the most recent date in the dataframe
        if date is None:
            date = self.returns_df.index[-1]

        # Extract the return window for the long sleeve
        returns_window_long = (
            self.returns_df[selected_stocks_long].loc[:date].tail(window)
        )

        returns_window_long_forcov = returns_window_long.loc[returns_window_long.ne(0).any(axis=1)]
        cov_long = returns_window_long_forcov.cov()  # covariance matrix for the long sleeve

        n_long = len(selected_stocks_long)

        if self.market_neutral:
            long_target = self.short_allocation
            short_target = self.short_allocation  
        else:
            long_target = 1.0 + self.short_allocation
            short_target = self.short_allocation 

        # ERC OBJECTIVE (relative contributions)
        def erc_objective_relative(w, cov):
            # Risk contribution:
            rc = w * (cov @ w)
            total_rc = np.sum(rc)
            # Relative contribution of each asset
            rel_rc = rc / total_rc
            # Target = 1 / n  ⇒ minimise the deviation from equal contributions
            n = len(w)
            target = 1 / n
            return np.sum((rel_rc - target) ** 2)

        # Equal-weight initial guess
        w0_long = np.ones(n_long) * (long_target / n_long)
        bounds_long = [(0, long_target)] * n_long
        cons_long = [{'type': 'eq', 'fun': lambda w: np.sum(w) - long_target}]

        res_long = minimize(
            erc_objective_relative,
            w0_long,
            args=(cov_long,),
            method='SLSQP',
            bounds=bounds_long,
            constraints=cons_long,
            options={'ftol': 1e-9, 'maxiter': 1000},
        )

        if not res_long.success:
            raise ValueError("ERC optimisation for long sleeve failed: " + res_long.message)

        w_long_opt = res_long.x
        portfolio_long = pd.DataFrame(w_long_opt, index=selected_stocks_long, columns=["Weight"])

        # SHORT SLEEVE
        if self.short_allocation != 0:
            # Optimise on positive intensities, then flip the sign
            returns_window_short = (
                self.returns_df[selected_stocks_short].loc[:date].tail(self.window)
            )

            returns_window_short_forcov = returns_window_short.loc[returns_window_short.ne(0).any(axis=1)]
            cov_short = returns_window_short_forcov.cov()

            n_short = len(selected_stocks_short)

            w0_short = np.ones(n_short) * (short_target / n_short)
            bounds_short = [(0, short_target)] * n_short
            cons_short = [{'type': 'eq', 'fun': lambda w: np.sum(w) - short_target}]

            res_short = minimize(
                erc_objective_relative,
                w0_short,
                args=(cov_short,),
                method='SLSQP',
                bounds=bounds_short,
                constraints=cons_short,
                options={'ftol': 1e-9, 'maxiter': 1000},
            )

            if not res_short.success:
                raise ValueError("ERC optimisation for short sleeve failed: " + res_short.message)

            # Convert positive intensities into negative weights
            w_short_opt = -res_short.x
            portfolio_short = pd.DataFrame(w_short_opt, index=selected_stocks_short, columns=["Weight"])
        else:
            portfolio_short = pd.DataFrame()

        # Combine long and short allocations
        portfolio_weights = pd.concat([portfolio_long, portfolio_short])

        return portfolio_weights

    def create_equal_weight_portfolio(self, start_date=None):
        """
        Build an equal-weighted (EW) portfolio based on *self.returns_df*.
        For each date, only the securities that are available (non-NaN)
        receive weight, distributed equally across those names.
        """

        # Determine the start date
        if start_date is None:
            start_date = self.returns_df.index[0]
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)

        # Keep returns from *start_date* onwards
        returns_filtered = self.returns_df.loc[start_date:]

        # Count how many assets are available (non-NaN) on each date
        count_available = returns_filtered.notna().sum(axis=1)

        # Initialise a DataFrame of equal weights for non-NaN assets
        # Cast to float (1.0) to simplify division
        weights_df = returns_filtered.notna().astype(float)

        # Divide each row by the number of available assets on that date
        weights_df = weights_df.div(count_available, axis=0).fillna(0)

        # Compute portfolio returns:
        # replace NaNs with 0 so missing assets do not contribute
        portfolio_returns = (returns_filtered.fillna(0) * weights_df).sum(axis=1)

        return weights_df, portfolio_returns






