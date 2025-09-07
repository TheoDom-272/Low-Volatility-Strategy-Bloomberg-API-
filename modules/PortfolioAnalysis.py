import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from dateutil.relativedelta import relativedelta
import webbrowser

class PortfolioStatistics:
    def __init__(self,
                 portfolio_returns: pd.Series,
                 benchmark_returns: pd.Series = None,
                 sectors_df: pd.DataFrame = None,
                 fees_series: pd.Series = None,
                 weights_df: pd.DataFrame = None,
                 bench_weights_df: pd.DataFrame = None,
                 port_long_returns = None,
                 port_short_returns = None,
                 asset_returns_df: pd.DataFrame = None,
                 all_returns_df: pd.DataFrame = None):

        self.portfolio = portfolio_returns.sort_index()
        non_zero = self.portfolio != 0
        self.portfolio = self.portfolio[non_zero]


        if asset_returns_df is not None:
            self.asset_returns_df = asset_returns_df.sort_index().loc[self.portfolio.index]
        else:
            self.asset_returns_df = None

        if benchmark_returns is not None:
            self.benchmark = benchmark_returns.sort_index().loc[self.portfolio.index]
        else:
            self.benchmark = None

        if port_long_returns is not None:
            self.port_long = port_long_returns.sort_index().loc[self.portfolio.index]
        else:
            self.port_long = None

        if port_short_returns is not None:
            self.port_short = port_short_returns.sort_index().loc[self.portfolio.index]
        else:
            self.port_short = None

        if fees_series is not None:
            self.fees_series = fees_series
        
        if all_returns_df is not None:
            self.all_returns_df = all_returns_df.sort_index().loc[self.portfolio.index]



        if sectors_df is None:
            self.sector_map = {}
        else:
            if isinstance(sectors_df, pd.DataFrame):
                series = sectors_df.iloc[:, 0]
            else:
                series = sectors_df
            self.sector_map = series.dropna().to_dict()

        self.weights_df = weights_df
        self.bench_weights_df = bench_weights_df
        self.stats = {'portfolio': {'performance': {}, 'risk': {}},
                      'benchmark': {'performance': {}, 'risk': {}}}
        self.plots_paths = {}
        self.graphs_dir = os.path.join(os.getcwd(), "Graphique")
        os.makedirs(self.graphs_dir, exist_ok=True)

    def _infer_periods_per_year(self, series: pd.Series) -> float:
        days = series.index.to_series().diff().dt.days.dropna().values
        median_days = np.median(days) if len(days) else 1
        per = 365.0 / median_days
        return float(min(max(per, 1), 252))

    def generate_plots(self):
        self._plot_cumulative_performance()
        if self.port_long is not None and self.port_short is not None:
            self._plot_cum_longshort()
            
        self._plot_drawdown()
        self._plot_sharpe_ratio()
        if self.benchmark is not None:
            self._plot_beta_evolution()
            self._plot_volatility_comparison()
        if self.fees_series is not None:
            plt.figure(figsize=(10,4))
            self.fees_series.plot(marker='o')
            plt.title('Frais de transaction')
            plt.xlabel('Date')
            plt.ylabel('Coût')
            plt.grid(True)
            p = os.path.join(self.graphs_dir, 'transaction_costs.png')
            plt.savefig(p, dpi=150); plt.close()
            self.plots_paths['transaction_costs'] = p

        # Plot sector allocations
        if self.weights_df is not None:
            self.plot_sector_allocations()

        # Plot relative sector exposure
        if self.bench_weights_df is not None:
            self.plot_relative_sector_exposure()

    def _plot_cumulative_performance(self):
        plt.figure(figsize=(10,6))
        (1+self.portfolio).cumprod().plot(label='Portefeuille')
        if self.benchmark is not None:
            (1+self.benchmark).cumprod().plot(label='Benchmark', ls='--')
        plt.title('Performance cumulative')
        plt.xlabel('Date'); plt.ylabel('Valeur'); plt.legend(); plt.grid(True)
        p = os.path.join(self.graphs_dir, 'performance.png')
        plt.savefig(p, dpi=150); plt.close()
        self.plots_paths['performance'] = p
    
    def _plot_cum_longshort(self):
        """
        Cumulative performance: benchmark + long sleeve + short sleeve
        """
        plt.figure(figsize=(10,6))

        # Portfolio
        (1 + self.portfolio).cumprod().plot(label='Portefeuille total', color='black', linewidth=2)

        # Benchmark
        if self.benchmark is not None:
            (1 + self.benchmark).cumprod().plot(label='Benchmark', ls='--')
        # Poche long
        if hasattr(self, 'port_long') and self.port_long is not None:
            (1 + self.port_long).cumprod().plot(label='Long sleeve', linestyle='-')
        # Poche short
        if hasattr(self, 'port_short') and self.port_short is not None:
            (1 + self.port_short).cumprod().plot(label='Short sleeve', linestyle=':')

        plt.title('Cumulative performance — Long & Short vs Benchmark')
        plt.xlabel('Date'); plt.ylabel('Value'); plt.legend(); plt.grid(True)

        p = os.path.join(self.graphs_dir, 'performance_longshort.png')
        plt.savefig(p, dpi=150)
        plt.close()

        self.plots_paths['performance_longshort'] = p


    def _plot_drawdown(self):
        cum = (1+self.portfolio).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        plt.figure(figsize=(10,6))
        dd.plot(color='red')
        plt.title('Drawdown historique')
        plt.xlabel('Date'); plt.ylabel('Drawdown'); plt.grid(True)
        p = os.path.join(self.graphs_dir, 'drawdown.png')
        plt.savefig(p, dpi=150); plt.close()
        self.plots_paths['drawdown'] = p

    def _plot_beta_evolution(self, window=60):
        df = pd.concat([self.portfolio.rename('port'), self.benchmark.rename('bench')], axis=1).dropna()
        roll_cov = df['port'].rolling(window).cov(df['bench'])
        roll_var = df['bench'].rolling(window).var()
        beta = roll_cov / roll_var
        plt.figure(figsize=(10,6))
        beta.plot(label=f'Beta {window}', color='blue')
        plt.axhline(1, ls='--', color='grey', label='Beta=1')
        plt.title(f'Beta Rolling sur {window} périodes')
        plt.xlabel('Date'); plt.ylabel('Beta'); plt.legend(); plt.grid(True)
        p = os.path.join(self.graphs_dir, 'rolling_beta.png')
        plt.savefig(p, dpi=150); plt.close()
        self.plots_paths['rolling_beta'] = p

    def _plot_volatility_comparison(self, window=60):
        per = self._infer_periods_per_year(self.portfolio)
        ann = np.sqrt(per)
        vol_p = self.portfolio.rolling(window).std() * ann
        vol_b = self.benchmark.rolling(window).std() * ann
        plt.figure(figsize=(10,6))
        vol_p.plot(label='Vol Portefeuille')
        vol_b.plot(label='Vol Benchmark', ls='--')
        plt.title(f'Volatilité mobile sur {window} périodes')
        plt.xlabel('Date'); plt.ylabel('Volatilité'); plt.legend(); plt.grid(True)
        p = os.path.join(self.graphs_dir, 'volatility_comparison.png')
        plt.savefig(p, dpi=150); plt.close()
        self.plots_paths['volatility_comparison'] = p

    def _plot_sharpe_ratio(self, window: int = 60):
        """
        Rolling Sharpe ratio (annualised, risk-free rate = 0) for the
        portfolio and – if provided – the benchmark.
        """
        per_year = self._infer_periods_per_year(self.portfolio)
        ann_factor = np.sqrt(per_year)

        # portfolio
        roll_mean_p = self.portfolio.rolling(window).mean()
        roll_std_p = self.portfolio.rolling(window).std()
        sharpe_p = (roll_mean_p / roll_std_p) * ann_factor

        # benchmark (if any)
        if self.benchmark is not None:
            roll_mean_b = self.benchmark.rolling(window).mean()
            roll_std_b = self.benchmark.rolling(window).std()
            sharpe_b = (roll_mean_b / roll_std_b) * ann_factor

        # plot
        plt.figure(figsize=(10, 6))
        sharpe_p.plot(label="Sharpe Portefeuille")
        if self.benchmark is not None:
            sharpe_b.plot(label="Sharpe Benchmark", linestyle="--")
        plt.title(f"Sharpe ratio annualisé – fenêtre mobile de {window} périodes")
        plt.xlabel("Date");
        plt.ylabel("Sharpe");
        plt.legend();
        plt.grid(True)

        p = os.path.join(self.graphs_dir, "sharpe_ratio.png")
        plt.savefig(p, dpi=150);
        plt.close()
        self.plots_paths["sharpe_ratio"] = p

    def calculate_sector_allocations(self):
        """
        Compute average sector allocation (%)
        separately for long positions, short positions, and total.
        """
        if self.weights_df is None:
            raise ValueError("weights_df not provided")

        # Compute average weight per ticker across all dates
        avg_w = self.weights_df.mean(axis=0)

        # Split into long and short positions
        long_w = avg_w[avg_w > 0]
        short_w = avg_w[avg_w < 0]

        # Build DataFrames to group by sector
        df_long = long_w.rename("weight").reset_index().rename(columns={"index": "ticker"})
        df_short = short_w.rename("weight").reset_index().rename(columns={"index": "ticker"})
        df_long["sector"] = df_long["ticker"].map(self.sector_map)
        df_short["sector"] = df_short["ticker"].map(self.sector_map)

        # Sum weights by sector and convert to percentages
        sector_long = df_long.groupby("sector")["weight"].sum() * 100
        sector_short = df_short.groupby("sector")["weight"].sum() * 100
        sector_all = sector_long.add(sector_short, fill_value=0)

        return sector_long, sector_short, sector_all

    def plot_sector_allocations(self):
        """
        Plot three bar charts showing average sector allocation:
        - long positions only
        - short positions only
        - combined long + short
        """
        sl, ss, sa = self.calculate_sector_allocations()
        for name, series in [("long", sl), ("short", ss), ("all", sa)]:
            if series is None or series.empty:
                print(f"[INFO] Pas de positions '{name}' à afficher.")
                continue

            fig, ax = plt.subplots(figsize=(8, 5))
            series.sort_values().plot(kind="bar", ax=ax)
            ax.set_title(f"Average sector allocation ({name})")
            ax.set_ylabel("Allocation (%)")
            ax.set_xlabel("Sector")
            # Annotate bars with percentage values
            for p in ax.patches:
                ax.annotate(
                    f"{p.get_height():.2f}%",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=8
                )
            path = os.path.join(self.graphs_dir, f"sector_alloc_{name}.png")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            self.plots_paths[f"sector_alloc_{name}"] = path

    def plot_relative_sector_exposure(self):
        """
        Plot a horizontal bar chart of relative sector exposure:
        (average portfolio allocation minus average benchmark allocation)
        """
        if self.bench_weights_df is None:
            raise ValueError("bench_weights_df not provided")

        # Compute average weights across time
        avg_w = self.weights_df.mean(axis=0)
        avg_b = self.bench_weights_df.mean(axis=0)

        # Build a DataFrame with portfolio and benchmark averages
        df = pd.DataFrame({
            "portfolio": avg_w,
            "benchmark": avg_b
        }).fillna(0)
        df["sector"] = df.index.map(self.sector_map)

        # Sum allocations by sector and convert to percentages
        port_sec = df.groupby("sector")["portfolio"].sum() * 100
        bench_sec = df.groupby("sector")["benchmark"].sum() * 100
        rel = (port_sec - bench_sec).sort_values()

        # Plot horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        rel.plot(kind="barh", ax=ax)
        ax.set_title("Relative sector exposure (portfolio − benchmark)")
        ax.set_xlabel("Relative allocation (%)")
        # Annotate bars with percentage values
        for p in ax.patches:
            ax.annotate(
                f"{p.get_width():.2f}%",
                (p.get_width(), p.get_y() + p.get_height() / 2),
                ha="left", va="center", fontsize=8
            )
        path = os.path.join(self.graphs_dir, "sector_exposure_relative.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        self.plots_paths["sector_exposure_relative"] = path

    def calculate_annualized_return(self, series: pd.Series, years: int) -> float:
        end = series.index[-1]
        start = end - relativedelta(years=years)
        sub = series.loc[series.index>=start]
        if len(sub)<2: return np.nan
        cum = (1+sub).prod() - 1
        return (1+cum)**(1/years) - 1

    def calculate_annualized_vol(self, series: pd.Series, years: int) -> float:
        end = series.index[-1]
        start = end - relativedelta(years=years)
        sub = series.loc[series.index>=start]
        if len(sub)<2: return np.nan
        n = len(sub)-1
        sd = sub.std()
        return sd * np.sqrt(n/years)

    def tracking_error_ex_post(self) -> float:
        """
        Ex-post tracking error vs. benchmark:
        the annualized standard deviation of (portfolio – benchmark) returns.
        """
        if self.benchmark is None:
            raise ValueError("Benchmark returns are required for ex-post tracking error.")
        diff = (self.portfolio - self.benchmark).dropna()
        # daily TE, then annualize
        te_daily = diff.std()
        per = self._infer_periods_per_year(self.portfolio)
        return te_daily * (per ** 0.5)

    def tracking_error_ex_ante(self) -> float:
        """
        Ex-ante tracking error vs. benchmark:
        sqrt[(w_p – w_b)' Σ (w_p – w_b)], annualized,
        where Σ is the asset-returns covariance matrix.
        Requires you pass in `asset_returns_df` when constructing.
        """
        if self.weights_df is None or self.bench_weights_df is None:
            raise ValueError("Both portfolio and benchmark weights are required for ex-ante TE.")
        if not hasattr(self, "asset_returns_df") or self.asset_returns_df is None:
            raise ValueError("Asset-level returns DataFrame (asset_returns_df) is required for ex-ante TE.")

        # 1) Covariance matrix of asset returns
        cov = self.asset_returns_df.cov()

        # 2) Average weight difference vector
        w_p = self.weights_df.mean(axis=0)
        w_b = self.bench_weights_df.mean(axis=0)
        w_diff = (w_p - w_b).reindex(cov.columns).fillna(0).values

        # 3) TE² = w_diff' Σ w_diff
        te2 = float(w_diff @ cov.values @ w_diff)

        # annualize
        per = self._infer_periods_per_year(self.portfolio)
        return (te2 ** 0.5) * (per ** 0.5)

    def calculate_statistics(self):
        # Portfolio stats
        py_p = self._infer_periods_per_year(self.portfolio)
        mu_p, sigma_p = self.portfolio.mean(), self.portfolio.std()
        z = stats.norm.ppf(0.05)

        last_date = self.portfolio.index[-1]
        w_p = self.weights_df.loc[last_date] 

        sharpe_p = (mu_p / sigma_p) * np.sqrt(py_p) if sigma_p != 0 else np.nan

        # VaR ex-ante portfolio
        asset_window  = self.asset_returns_df.loc[:last_date]
        var_mc_ex_ante = self.calculate_var_monte_carlo(w_p, asset_window)

        

        perf_p = {
            'Cumulative Return': (1+self.portfolio).prod()-1,
            'Annualized Return (T.)': (1+self.portfolio).prod()**(py_p/len(self.portfolio)) -1,
            '1Y Return (ann.)': self.calculate_annualized_return(self.portfolio,1),
            '3Y Return (ann.)': self.calculate_annualized_return(self.portfolio,3),
            '5Y Return (ann.)': self.calculate_annualized_return(self.portfolio,5),
            'Sharpe Ratio (ann.)': round(sharpe_p,2),
            'Best Month':  round(self.portfolio.max()   * 100, 2),
            'Worst Month': round(self.portfolio.min()   * 100, 2),
            'Win Rate':    round((self.portfolio>0).mean() * 100, 2),
        }

        risk_p = {
            'Volatility': sigma_p * np.sqrt(py_p),
            '1Y Volatility': self.calculate_annualized_vol(self.portfolio,1),
            '3Y Volatility': self.calculate_annualized_vol(self.portfolio,3),
            '5Y Volatility': self.calculate_annualized_vol(self.portfolio,5),
            'Max Drawdown': self.calculate_max_drawdown(self.portfolio),
            'VaR 95% (Hist.)': -np.percentile(self.portfolio,5)*np.sqrt(py_p),
            'VaR 95% (Param.)': -(mu_p + sigma_p*z)*np.sqrt(py_p),
            'VaR 95% (MC)': var_mc_ex_ante*np.sqrt(py_p),
            'Skewness': stats.skew(self.portfolio),
            'Kurtosis': stats.kurtosis(self.portfolio)
        }
        self.stats['portfolio']['performance'], self.stats['portfolio']['risk'] = perf_p, risk_p

        # Benchmark stats
        if self.benchmark is not None:
            py_b = self._infer_periods_per_year(self.benchmark)
            mu_b, sigma_b = self.benchmark.mean(), self.benchmark.std()

            sharpe_b = (mu_b / sigma_b) * np.sqrt(py_b) if sigma_b != 0 else np.nan  

            w_b = self.bench_weights_df.loc[last_date]
            asset_rets_full = self.asset_returns_df.loc[:last_date]

            var_mc_bench_ex_ante = self.calculate_var_monte_carlo(w_b, asset_rets_full)

            perf_b = {
                'Cumulative Return': (1+self.benchmark).prod()-1,
                'Annualized Return (T.)': (1+self.benchmark).prod()**(py_b/len(self.benchmark)) -1,
                '1Y Return (ann.)': self.calculate_annualized_return(self.benchmark,1),
                '3Y Return (ann.)': self.calculate_annualized_return(self.benchmark,3),
                '5Y Return (ann.)': self.calculate_annualized_return(self.benchmark,5),
                'Sharpe Ratio (ann.)': round(sharpe_b,2),  
                'Best Month': round(self.benchmark.max()*100,2),
                'Worst Month': round(self.benchmark.min()*100,2),
                'Win Rate': round((self.benchmark>0).mean()*100,2)
            }
            risk_b = {
                'Volatility': sigma_b * np.sqrt(py_b),
                '1Y Volatility': self.calculate_annualized_vol(self.benchmark,1),
                '3Y Volatility': self.calculate_annualized_vol(self.benchmark,3),
                '5Y Volatility': self.calculate_annualized_vol(self.benchmark,5),
                'Max Drawdown': self.calculate_max_drawdown(self.benchmark),
                'VaR 95% (Hist.)': -np.percentile(self.benchmark,5)*np.sqrt(py_b),
                'VaR 95% (Param.)': -(mu_b + sigma_b*z)*np.sqrt(py_b),
                'VaR 95% (MC)': var_mc_bench_ex_ante*np.sqrt(py_b),
                'Skewness': stats.skew(self.benchmark),
                'Kurtosis': stats.kurtosis(self.benchmark)
            }
            self.stats['benchmark']['performance'], self.stats['benchmark']['risk'] = perf_b, risk_b

    def generate_html_report(self, filename='Portfolio_Report.html'):
        self.calculate_statistics()
        result_dir = os.path.join(os.path.dirname(os.getcwd()), "Result")
        os.makedirs(result_dir, exist_ok=True)
        output_path = os.path.join(result_dir, filename)

        perf_p = self.stats['portfolio']['performance']
        risk_p = self.stats['portfolio']['risk']
        perf_b = self.stats['benchmark']['performance']
        risk_b = self.stats['benchmark']['risk']

        # Construire deux tableaux côte à côte
        html = ["<!DOCTYPE html><html lang='fr'><head><meta charset='UTF-8'><title>Rapport d'Analyse</title>"
                "<style>table{border-collapse:collapse;}td,th{border:1px solid #ccc;padding:8px;}</style></head><body>"]
        html.append("<h1>Rapport d'Analyse de Portfolio</h1>")
        html.append("<div style='display:flex;gap:40px;'>")
        # Performance
        html.append("<div><h2>Rendements</h2><table><tr><th>Indicateur</th><th>Portefeuille</th><th>Benchmark</th></tr>")
        for key,val in perf_p.items():
            fmt = f"{val:.2%}" if isinstance(val,(float,np.floating)) and 'Return' in key else f"{val:.2%}" if 'Return' in key else f"{val:.2%}" if '%' in key else str(val)
            fb = perf_b[key]
            fmtb = f"{fb:.2%}" if isinstance(fb,(float,np.floating)) and 'Return' in key else str(fb)
            html.append(f"<tr><td>{key}</td><td>{fmt}</td><td>{fmtb}</td></tr>")
        html.append("</table></div>")
        # Risque
        html.append("<div><h2>Indicateurs de Risque</h2><table><tr><th>Indicateur</th><th>Portefeuille</th><th>Benchmark</th></tr>")
        for key,val in risk_p.items():
            fmt = f"{val:.2%}" if 'Volatility' in key or 'Drawdown' in key or 'VaR' in key else f"{val:.2f}"
            fb = risk_b[key]
            fmtb = f"{fb:.2%}" if 'Volatility' in key or 'Drawdown' in key or 'VaR' in key else f"{fb:.2f}"
            html.append(f"<tr><td>{key}</td><td>{fmt}</td><td>{fmtb}</td></tr>")
        html.append("</table></div>")
        html.append("</div><hr>")
        # Graphiques
        for title, key in [
            ("Performance cumulative", "performance"),
            ("Performance Long/Short vs Benchmark", "performance_longshort"),
            ("Drawdown historique", "drawdown"),
            ("Beta Rolling", "rolling_beta"),
            ("Volatilité comparée", "volatility_comparison"),
            ("Sharpe ratio", "sharpe_ratio"),
            ("Frais", "transaction_costs"),
            ("Sector Allocation (long)", "sector_alloc_long"),
            ("Sector Allocation (short)", "sector_alloc_short"),
            ("Sector Allocation (total)", "sector_alloc_all"),
            ("Relative Sector Exposure", "sector_exposure_relative"),
        ]:
            if key in self.plots_paths:
                html.append(f"<h3>{title}</h3><img src='{self.plots_paths[key]}' width='800'><br>")
        html.append("</body></html>")

        with open(output_path,'w',encoding='utf-8') as f:
            f.write(''.join(html))
        # Ouvrir explicitement avec Edge
        edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        webbrowser.register('edge', None, webbrowser.BackgroundBrowser(edge_path))
        webbrowser.get('edge').open('file://' + os.path.abspath(output_path))
        print(f"Rapport HTML généré et ouvert : {output_path}")

    @staticmethod
    def calculate_max_drawdown(series: pd.Series) -> float:
        cum = (1+series).cumprod()
        return ((cum-cum.cummax())/cum.cummax()).min()

    def calculate_var_monte_carlo_last(self, weights: pd.Series, per_year: float, samples: int = 100_000) -> float:
        """
        VaR 95% Monte-Carlo multivariée ex-ante :
        """

        rets = self.asset_returns_df.loc[:, weights.index].dropna(how='all')

        mu  = rets.mean().values * per_year
        cov = rets.cov().values  * per_year

        epsilon = 1e-8
        max_tries = 10
        for i in range(max_tries):
            try:
                L = np.linalg.cholesky(cov + epsilon * np.eye(cov.shape[0]))
                break
            except np.linalg.LinAlgError:
                epsilon *= 10
        else:
            raise np.linalg.LinAlgError(
                f"Covariance matrix not PD even after regularization up to {epsilon:.1e}"
            )

        z    = np.random.standard_normal((cov.shape[0], samples))
        sims = mu[:, None] + L @ z
        pl = weights.values @ sims

        return -np.percentile(pl, 5)
    
    def calculate_var_monte_carlo(self,weights: pd.Series,returns: pd.DataFrame,half_life: int = 21,horizon_days: int = 1,alpha: float = 0.95,n_mc: int = 20_000) -> float:

        """
        Bloomberg-style MC-VaR with EWMA volatilities.
        """

        invested = weights.loc[weights.abs() > 0]               # poids ≠ 0
        common   = returns.columns.intersection(invested.index) 
        if common.empty:
            return np.nan
        
        # Sous-ensemble poids / rendements
        w = invested.reindex(common).fillna(0)
        rets = returns[common]

        # Supprimer les colonnes plates (variance nulle ou tout à 0/NaN)
        mask_valid = (rets != 0).any() & rets.notna().any()
        rets = rets.loc[:, mask_valid]
        w    = w.reindex(rets.columns).fillna(0)

        # Après nettoyage, assez d’actifs ?
        if rets.shape[1] == 0 or w.abs().sum() == 0:
            return np.nan

        # Supprimer les lignes à NaN
        rets = rets.dropna(how='any')
        if len(rets) < 2:
            return np.nan

        # EWMA covariance (daily)
        cov_d = self.ewma_cov(rets, half_life).values

        # 2Scale to chosen horizon (√h) for covariance, ×h for mean
        cov_h = cov_d * horizon_days
        mu_h  = rets.mean().values * horizon_days

        # Cholesky (with tiny ridge to ensure PD)
        ridge = 1e-8
        for _ in range(10):
            try:
                L = np.linalg.cholesky(cov_h + ridge * np.eye(cov_h.shape[0]))
                break
            except np.linalg.LinAlgError:
                ridge *= 10
        else:
            return np.nan  
        #L = np.linalg.cholesky(cov_h + 1e-8 * np.eye(cov_h.shape[0]))

        # Simulate correlated shocks
        z    = np.random.randn(cov_h.shape[0], n_mc)
        sims = mu_h[:, None] + L @ z            # (n_assets × n_mc)

        # Portfolio P&L
        pnl = w.values @ sims                          # shape (n_mc,)

        # VaR
        perc = 100 * (1 - alpha)
        return -np.percentile(pnl, perc)
    

    def ewma_cov(self,returns: pd.DataFrame, half_life: int) -> pd.DataFrame:
        """
        Compute the exponentially-weighted covariance matrix
        using the standard RiskMetrics / Bloomberg formulation.
        """
        # λ such that weight = 0.5 after `half_life` observations
        lam = 0.5 ** (1 / half_life)

        # demeaned returns (necessary for covariance)
        x = returns - returns.mean()

        # EWMA weights
        w = (1 - lam) * lam ** np.arange(len(x) - 1, -1, -1)
        w /= w.sum()                                   # normalise to 1

        # apply weights
        weighted = (x.T * w).T
        cov = weighted.T @ x                           # (n × T)·(T × n) = (n × n)
        return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)






