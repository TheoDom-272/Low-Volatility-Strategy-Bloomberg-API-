import pandas as pd
import numpy as np

from InvestmentStrategies import PortfolioManagement
from Data_Management import DataExport


class Backtester() :

    def __init__(self,all_betas_df, all_prices_df, all_returns_df, all_volumes_df,start_date,method_optim,short_allocation, n_freq_rebal=5,n_freq_selection =20,
                        min_stock = 0.15, beta_threshold=0.20, volume_threshold=-1, window=126,tc_bps=0,composition_df=None,market_neutral=False):
        self.all_betas_df = all_betas_df
        self.all_prices_df = all_prices_df
        self.all_returns_df = all_returns_df
        self.all_volumes_df = all_volumes_df
        self.n_freq_rebal = n_freq_rebal
        self.n_freq_selection = n_freq_selection
        self.volume_threshold = volume_threshold
        self.window = window
        self.min_stock = min_stock
        self.beta_threshold = beta_threshold
        self.start_date = start_date
        self.method_optim = method_optim
        self.short_allocation = short_allocation
        self.tc_bps = tc_bps
        self.composition_df = composition_df
        self.market_neutral=market_neutral


    def compute_selection_interval(self, rebal_freq, selection_freq):
        """
        Compute how many rebalancing dates occur **between** two stock-selection
        events, given the chosen rebalancing and selection frequencies.

        Possible frequencies are: "Weekly", "Monthly", "Quarterly", "Annually".
        """

        # Map each label to its number of occurrences per calendar year
        freq_map = {"Weekly": 52, "Monthly": 12, "Quarterly": 4, "Annually": 1}

        rebal_per_year = freq_map[rebal_freq]
        selection_per_year = freq_map[selection_freq]

        # Round to the nearest integer to get an easy-to-use ratio
        ratio = int(round(rebal_per_year / selection_per_year))

        # Guarantee a minimum interval of one rebalance
        if ratio < 1:
            ratio = 1

        return ratio

    def get_resample_rule(self, freq_str):
        mapping = {"Weekly": "W-FRI", "Monthly": "ME", "Quarterly": "QE", "Annually": "A"}
        return mapping.get(freq_str, "M")

    def get_window_days(self):
        """
        Return the number of trading days corresponding to ``self.n_freq_selection``.

        Default mapping
        ---------------
          - "Weekly"    : ~5 trading days
          - "Monthly"   : ~21 trading days
          - "Quarterly" : 63 trading days (252 / 4)
          - "Annually"  : 252 trading days
        """
        mapping = {
            "Weekly": 5,
            "Monthly": 21,
            "Quarterly": 63,
            "Annually": 252
        }

        if self.n_freq_selection in mapping:
            return mapping[self.n_freq_selection]
        else:
            raise ValueError(
                f"n_freq_selection '{self.n_freq_selection}' not recognised. "
                f"Allowed values are: {list(mapping.keys())}."
            )



    def Backtest(self):
        """
        Run a back-test with daily weight drift.

        • Rebalancing / (re)selection dates are still driven by
          n_freq_rebal and n_freq_selection.

        • Between two rebalancings the weights follow:
              w_t = w_{t-1} * (1 + r_t),
          then, optionally, are renormalised so that Σ w_t = 1.

        • Transaction costs (tc_bps) are applied *only* on
          rebalancing dates.
        """

        # ---------- Data preparation ----------
        returns = self.all_returns_df.copy()  # daily returns
        returns.index = pd.to_datetime(returns.index)
        tickers = returns.columns
        dates = returns.index  # trading-day master index

        # ---------- First rebalancing date ----------
        start_dt = pd.to_datetime(self.start_date)
        if start_dt not in dates:  # align on first trading day ≥ start_dt
            pos = dates.searchsorted(start_dt)
            if pos == len(dates):
                raise ValueError("start_date is after the last available trading day.")
            start_dt = dates[pos]

        # ---------- Build the rebalancing schedule ----------
        rebal_rule = self.get_resample_rule(self.n_freq_rebal)

        # (1) “Nominal” labels produced by resample (may fall on weekends / holidays)
        raw_labels = (
            returns.loc[start_dt:].resample(rebal_rule).last().index  # index of labels
        )

        # (2) Align each label on the last *trading* day ≤ label
        aligned = []
        for lab in raw_labels:
            pos = dates.searchsorted(lab, side="right") - 1
            if pos >= 0:
                aligned.append(dates[pos])

        rebal_dates = pd.DatetimeIndex(aligned).unique().sort_values()

        # Force the very first label to be start_dt
        if rebal_dates[0] != start_dt:
            rebal_dates = pd.DatetimeIndex([start_dt]).append(rebal_dates)

        # ---------- Selection / optimisation frequency ----------
        selection_interval = self.compute_selection_interval(
            self.n_freq_rebal, self.n_freq_selection
        )
        print(f"\n=> New universe selection every {selection_interval} rebal. periods.")

        # ---------- Output containers ----------
        weights_daily_df = pd.DataFrame(index=dates, columns=tickers, dtype=float)
        rebal_weights_df = pd.DataFrame(index=rebal_dates, columns=tickers, dtype=float)
        fees_series = pd.Series(dtype=float, index=rebal_dates)
        portfolio_returns = pd.Series(dtype=float, index=dates)
        Long_returns = pd.Series(dtype=float, index=dates)
       
        bench_weights_df = pd.DataFrame(0.0, index=dates, columns=tickers)
        bench_returns = pd.Series(index=dates, dtype=float)

        if self.short_allocation > 0 :
            short_returns = pd.Series(dtype=float, index=dates)
        else:
            short_returns = None

        # ---------- Main loop ----------
        current_weights = pd.Series(0.0, index=tickers)  # weights *before* first rebal
        selected_long = []
        selected_short = []
        last_selection_dt = None
        tc_rate = self.tc_bps / 10_000

        current_year = None
        current_bench_weights = None

        for idx, rebal_dt in enumerate(rebal_dates):


            # BENCHMARK ANNUAL REBALANCE
            # At each rebalancing step, check if year changed
            year = rebal_dt.year
            if year != current_year:
                # New calendar year: build equal-weight vector on that year's constituents
                current_year = year

                if self.composition_df is not None and year in self.composition_df.columns:
                    # get tickers for this year
                    year_list = self.composition_df.loc[:, year]
                    # keep only tickers actually in returns DataFrame
                    active = [t for t in year_list if t in tickers]
                else:
                    active = list(tickers)

                # if no constituents, fallback to full universe
                if not active:
                    active = list(tickers)

                # assign equal weights
                w = 1.0 / len(active)
                current_bench_weights = pd.Series(0.0, index=tickers)
                current_bench_weights.loc[active] = w

            # Fill benchmark weights for this rebalancing date
            bench_weights_df.loc[rebal_dt, :] = current_bench_weights.values

            # Now propagate daily between this rebal date and next rebal date
            if idx < len(rebal_dates) - 1:
                next_dt = rebal_dates[idx + 1]
                mask = (dates > rebal_dt) & (dates <= next_dt)
            else:
                mask = dates > rebal_dt

            # Fill the benchmark weights for the current rebalancing window
            bench_array = current_bench_weights.values
        
            bench_weights_df.loc[mask, :] = bench_array


            # Update selection schedule
            if idx % selection_interval == 0:
                print(f"  > Selection performed on: {rebal_dt}")

                # Dynamically update the universe based on annual composition
                year = rebal_dt.year
                if self.composition_df is not None and year in self.composition_df.columns:

                    # Retrieve list of tickers for this year from the "Tickers" column
                    tickers_year = self.composition_df.loc[:, year]
                    valid_tickers = [t for t in tickers_year if t in self.all_prices_df.columns]
                    valid = set(valid_tickers)
                    existing_beta   = [t for t in valid if t in self.all_betas_df.columns]
                    existing_price  = [t for t in valid if t in self.all_prices_df.columns]
                    existing_volume = [t for t in valid if t in self.all_volumes_df.columns]
                    existing_rets   = [t for t in valid if t in self.all_returns_df.columns]

                    print(f"new universe for {rebal_dt} with {len(existing_beta)} tickers.")

                    pm_view = PortfolioManagement(
                        prices_df  = self.all_prices_df[existing_price],
                        beta_df    = self.all_betas_df[existing_beta],
                        volume_df  = self.all_volumes_df[existing_volume],
                        returns_df = self.all_returns_df[existing_rets],
                        method_optim      = self.method_optim,
                        short_allocation  = self.short_allocation,
                        min_stock         = self.min_stock,
                        beta_threshold    = self.beta_threshold,
                        volume_threshold  = self.volume_threshold,
                        market_neutral    = self.market_neutral,)

            
                else:
                    print(f"[WARN] No composition available for year {year}, using full universe.")

                # Perform stock selection on the filtered universe
                selected_long, selected_short = pm_view.select_stocks(
                    date=rebal_dt,
                    window=self.get_window_days()
                )
            else:
                # No new selection this iteration; keep previous selection
                print(f"  > No new selection at {rebal_dt}")

            try :
                # Portfolio optimisation
                if self.method_optim == "Mean/Variance":
                    opt_weights = pm_view.optimize_portfolio_mv(
                        selected_stocks_long=selected_long,
                        selected_stocks_short=selected_short,
                        actual_weights=current_weights,
                        date=rebal_dt,
                    )
                else:  # Equal-Risk-Contribution
                    opt_weights = pm_view.optimize_portfolio_erc(
                        selected_stocks_long=selected_long,
                        selected_stocks_short=selected_short,
                        actual_weights=current_weights,
                        date=rebal_dt,
                        window=self.get_window_days(),
                    )
        
                target_weights = pd.Series(0.0, index=tickers)
                target_weights.update(opt_weights["Weight"])

                # Transaction costs
                turnover = (target_weights - current_weights).abs().sum()
                fees_series.loc[rebal_dt] = turnover * tc_rate

            except ValueError as e:
                 print(f"[WARN] {e}  — Keeping previous weights (no rebalance at {rebal_dt}).")

                 target_weights = current_weights.copy()
                 fees_series.loc[rebal_dt] = 0.0

            # Save rebalancing weights
            rebal_weights_df.loc[rebal_dt] = target_weights
            current_weights = target_weights.copy()  # state at *beginning* of interval

            # Daily drift until next rebalancing
            if idx < len(rebal_dates) - 1:
                window_dates = dates[(dates > rebal_dt) & (dates <= rebal_dates[idx + 1])]
            else:
                window_dates = dates[dates > rebal_dt]

            if len(window_dates) == 0:
                continue  # nothing to drift

            rets_win = returns.loc[window_dates]  # DataFrame (d_j, n_assets)
            gross_win = 1.0 + rets_win.values  # ndarray
            cumfactor = np.cumprod(gross_win, axis=0)  # cumulative drift factors

            w0 = current_weights.values[None, :]  # (1, n_assets)
            weights_raw = w0 * cumfactor  # buy-and-hold weights
            weights_raw = np.nan_to_num(weights_raw, nan=0.0)

            # Raw portfolio P&L before normalization
            #wealth = weights_raw.sum(axis=1)
            #port_rets = wealth[1:] / wealth[:-1] - 1
            port_rets = np.nansum(weights_raw[:-1] * rets_win.values[1:], axis=1)

            # Separate long vs. short normalization for weight tracking
            pos_mask = weights_raw > 0
            neg_mask = weights_raw < 0
            
            expo_long = np.sum(weights_raw * pos_mask, axis=1)[:-1]
            expo_short = -np.sum(weights_raw * neg_mask, axis=1)[:-1]

            port_rets_long = np.where(expo_long != 0,np.nansum((weights_raw * pos_mask)[:-1] * rets_win.values[1:], axis=1) / expo_long,0.0) 

            #port_rets_long = sum_pos[1:] / sum_pos[:-1] - 1
            if self.short_allocation > 0:
                port_rets_short = np.where(expo_short != 0,np.nansum((weights_raw * neg_mask)[:-1] * rets_win.values[1:], axis=1) / expo_short,0.0)
            else:
                port_rets_short = None

            # Initialize normalized weights
            weights_norm = np.zeros_like(weights_raw)

            sum_pos = (weights_raw * pos_mask).sum(axis=1)  # shape (T,)
            sum_neg = (-weights_raw * neg_mask).sum(axis=1)
            
            # Compute normalized weights
            idx_long = sum_pos != 0
            weights_norm[idx_long, :] = (weights_raw[idx_long, :] * pos_mask[idx_long, :]) / sum_pos[idx_long, None]
                                        
            idx_short = sum_neg != 0
            weights_norm[idx_short, :] = (weights_raw[idx_short, :] * neg_mask[idx_short, :]) / sum_neg[idx_short, None]

                                        

            # Inject into outputs
            weights_daily_df.loc[window_dates] = weights_norm
            portfolio_returns.loc[window_dates[1:]] = port_rets
            Long_returns.loc[window_dates[1:]] = port_rets_long
            if port_rets_short is not None:
                short_returns.loc[window_dates[1:]] = port_rets_short

        #  Final cleaning / forward-fill
        weights_daily_df = weights_daily_df.ffill().fillna(0)

        # compute benchmark returns on rebalancing window
        bench_returns = (returns * bench_weights_df).sum(axis=1)

        # Truncate all outputs so they start à start_dt
        mask = dates >= start_dt

   
        portfolio_returns = portfolio_returns.loc[mask]
        Long_returns   = Long_returns.loc[mask]

        if short_returns is not None:
            short_returns = short_returns.loc[mask]
        weights_daily_df  = weights_daily_df.loc[mask]
        bench_weights_df  = bench_weights_df.loc[mask]
        rebal_weights_df  = rebal_weights_df.loc[rebal_weights_df.index >= start_dt]
        fees_series       = fees_series.loc[fees_series.index      >= start_dt]
        bench_returns     = bench_returns.loc[mask]

        return (
            portfolio_returns.dropna(),  # daily portfolio return series
            weights_daily_df,
            rebal_weights_df,
            bench_weights_df,
            bench_returns,
            fees_series,
            Long_returns,
            short_returns
        )







