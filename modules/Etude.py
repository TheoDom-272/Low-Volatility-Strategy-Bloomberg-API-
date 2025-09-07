"""
This script gathers several exploratory analyses carried out during the
development phase:

    • validation of beta calculations (realised, shrinkage, DCC);
    • testing of the cross-sectional liquidity filter;
    • performance measurements and visual comparisons of the different methods.

These modules are not part of the final back-test, they serve only as a
toolbox to cross-check and validate the computational building blocks before
integrating them into the full strategy.
"""


from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data_Management import DataAPI
from InvestmentStrategies import BetaCalculator, PortfolioManagement

__all__ = [
    "demo_realized_vs_shrinkage",
    "demo_liquidity_filter",
    "demo_compare_betas",
]


def _load_index_data(
    index_name: str = "S&P 500",
    start_date: str = "2020-01-03",
    end_date: str = "2024-12-31",
) -> Dict[str, pd.DataFrame]:
    """Internal helper — fetches prices, returns, volumes… for *one* index."""

    api_instance = DataAPI(
        ticker=[index_name],
        start_date=start_date,
        end_date=end_date,
        api_name="Data Save",  # uses the local cache if available
    )
    return api_instance.dict_data[index_name]


# ---------------------------------------------------------------------------
# 1) Realised vs Shrinkage beta demo
# ---------------------------------------------------------------------------

def demo_realized_vs_shrinkage(
    index_name: str = "S&P 500",
    start_date: str = "2020-01-03",
    end_date: str = "2024-12-31",
    n_assets: int = 5,
    window: int = 90,
    prior_window: int = 252,
    annual_window: int = 252,
    seed: int = 42,
    show_plots: bool = True,
) -> Dict[str, float]:
    """Compare execution time + plots for realised beta and shrinkage beta.
    """


    # 1. Data loading and subsampling
    data = _load_index_data(index_name, start_date, end_date)
    stock_rets: pd.DataFrame = data["aligned_stock_returns"]
    market_rets: pd.Series | pd.DataFrame = data["aligned_market_returns"]

    # Pick n_assets random stocks for a cleaner figure
    rng = random.Random(seed)
    assets: List[str] = rng.sample(list(stock_rets.columns), k=min(n_assets, stock_rets.shape[1]))
    stock_rets = stock_rets[assets]

    calc = BetaCalculator()


    # 2. Non‑vectorised realised beta (loop) – reference implementation
    t0 = time.time()
    beta_dict_loop: Dict[str, pd.Series] = {}
    for asset in assets:
        beta_dict_loop[asset] = calc.calculate_realized_beta(
            stock_rets[asset], market_rets, asset, window=window
        )
    beta_df_loop = pd.DataFrame(beta_dict_loop)
    t_loop = time.time() - t0


    # 3. Vectorised realised beta
    t0 = time.time()
    beta_df_vec = calc.realized_beta_vectorized(stock_rets, market_rets, window=window)
    t_vec = time.time() - t0


    # 4. Shrinkage beta – loop then vectorised
    t0 = time.time()
    shrink_dict_loop = {}
    for asset in assets:
        shrink_dict_loop[asset] = calc.calculate_beta_shrinkage(
            stock_rets[asset], market_rets,
            window=window, prior_window=prior_window, annual_window=annual_window,
        )
    shrink_df_loop = pd.DataFrame(shrink_dict_loop)
    t_shrink_loop = time.time() - t0

    t0 = time.time()
    shrink_df_vec = calc.beta_shrinkage_vectorized(
        stock_rets, market_rets,
        short_window=window, prior_window=prior_window, annual_window=annual_window,
    )
    t_shrink_vec = time.time() - t0


    # 5. Plotting
    if show_plots:
        # Realised beta
        fig, axes = plt.subplots(nrows=len(assets), figsize=(10, 2.5 * len(assets)), sharex=True)
        if len(assets) == 1:
            axes = [axes]
        for ax, asset in zip(axes, assets):
            beta_df_loop[asset].plot(ax=ax, label="Loop", color="C0")
            beta_df_vec[asset].plot(ax=ax, label="Vectorised", color="C1", linestyle="--")
            ax.set_title(f"Realised beta – {asset}")
            ax.legend()
        plt.tight_layout()
        plt.show()

        # Shrinkage beta
        fig, axes = plt.subplots(nrows=len(assets), figsize=(10, 2.5 * len(assets)), sharex=True)
        if len(assets) == 1:
            axes = [axes]
        for ax, asset in zip(axes, assets):
            shrink_df_loop[asset].plot(ax=ax, label="Loop", color="C2")
            shrink_df_vec[asset].plot(ax=ax, label="Vectorised", color="C3", linestyle="--")
            ax.set_title(f"Shrinkage beta – {asset}")
            ax.legend()
        plt.tight_layout()
        plt.show()

    return {
        "realised_loop": t_loop,
        "realised_vectorised": t_vec,
        "shrink_loop": t_shrink_loop,
        "shrink_vectorised": t_shrink_vec,
    }



# 2) Cross‑sectional liquidity filter demo
def demo_liquidity_filter(
    index_name: str = "Russel 1000",
    start_date: str = "2020-01-03",
    end_date: str = "2024-12-31",
    window_days: int = 20,
    show_plots: bool = True,
):
    """Run the cross‑sectional liquidity filter on *one* date (last available)."""

    data = _load_index_data(index_name, start_date, end_date)
    prices_df: pd.DataFrame = data["prices"]
    volumes_df: pd.DataFrame = data["volumes"]
    returns_df: pd.DataFrame = data["returns"]

    # Dummy placeholders for attributes we do not use in this demo
    port = PortfolioManagement(
        prices_df=prices_df,
        beta_df=returns_df,
        volume_df=volumes_df,
        returns_df=returns_df,
        method_optim=None,
    )

    test_date = prices_df.index[-1]
    final_assets, excluded_assets = port.cross_sectional_liquidity_filter_with_history(
        prices_df, volumes_df, test_date, window_days=window_days,compare_log=True
    )

    if show_plots:
        df = pd.concat([
            pd.DataFrame({
                'asset': final_assets.index,
                'z': final_assets.values,
                'status': 'Kept'
            }),
            pd.DataFrame({
                'asset': excluded_assets.index,
                'z': excluded_assets.values,
                'status': 'Excluded'
            })
        ])

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        df['x'] = df.index

        plt.figure(figsize=(10, 6))
        mask_kept = df['status'] == 'Kept'
        plt.scatter(df.loc[mask_kept, 'x'],
                    df.loc[mask_kept, 'z'],
                    label='Kept',
                    color='C0',
                    alpha=0.7)
        mask_exc = df['status'] == 'Excluded'
        plt.scatter(df.loc[mask_exc, 'x'],
                    df.loc[mask_exc, 'z'],
                    label='Excluded',
                    color='C2',
                    marker='x',
                    alpha=0.7)

        plt.axhline(-1.96, color='red', linestyle='--', label='Threshold -1.96·MAD')
        plt.title(f"Liquidity z-scores on {test_date.date()}")
        plt.ylabel("Liquidity z-score")
        plt.xlabel("Assets (shuffled order)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 1. Retrieve yearly composition and isolate tickers for 2024
    comp_df = data.get("composition")           # DataFrame where columns = years
    if comp_df is not None and 2024 in comp_df.columns:
        tickers_2024 = comp_df[2024].dropna().tolist()
        tickers_2024 = [t for t in tickers_2024 if t in prices_df.columns]  # keep tickers with data
    else:
        tickers_2024 = []   # fallback → empty list

    if tickers_2024:
        # 2. Sub-dataframes restricted to 2024 universe
        prices_24  = prices_df[tickers_2024]
        volumes_24 = volumes_df[tickers_2024]

        # 3. Apply the same liquidity filter on the same test_date
        kept_24, excl_24 = port.cross_sectional_liquidity_filter_with_history(
            prices_24, volumes_24, test_date, window_days=window_days, compare_log=False
        )

        if show_plots:
            # small helper to plot kept vs excluded
            df_24 = pd.concat([
                pd.DataFrame({"asset": kept_24.index,  "z": kept_24.values,  "status": "Kept"}),
                pd.DataFrame({"asset": excl_24.index,  "z": excl_24.values,  "status": "Excluded"})
            ])
            df_24 = df_24.sample(frac=1, random_state=42).reset_index(drop=True)
            df_24["x"] = df_24.index

            plt.figure(figsize=(10, 6))
            m_keep = df_24["status"] == "Kept"
            plt.scatter(df_24.loc[m_keep, "x"],  df_24.loc[m_keep, "z"],
                        label="Kept (2024)", color="C0", alpha=0.7)
            m_exc  = df_24["status"] == "Excluded"
            plt.scatter(df_24.loc[m_exc, "x"],   df_24.loc[m_exc, "z"],
                        label="Excluded (2024)", color="C2", marker="x", alpha=0.7)

            plt.axhline(-1.96, color="red", ls="--", label="Threshold -1.96·MAD")
            plt.title(f"Liquidity z-scores on {test_date.date()} — 2024 constituents")
            plt.ylabel("Liquidity z-score")
            plt.xlabel("Assets (shuffled order)")
            plt.legend()
            plt.tight_layout()
            plt.show()

        print(f"[2024] kept: {len(kept_24)}, excluded: {len(excl_24)}")
    else:
        print("[WARN] No composition available for 2024; skipping restricted liquidity check.")

    return final_assets, excluded_assets



# 3) Realised vs Shrinkage vs DCC beta demo

def demo_compare_betas(
    index_name: str = "S&P 500",
    start_date: str = "2020-01-03",
    end_date: str = "2024-12-31",
    n_assets: int = 4,
    seed: int = 123,
    window: int = 90,
    prior_window: int = 252,
    annual_window: int = 252,
    show_plots: bool = True,
):
    """Overlay realised, shrinkage and DCC betas for n_assets random stocks."""

    data = _load_index_data(index_name, start_date, end_date)
    stock_rets = data["aligned_stock_returns"]
    market_rets = data["aligned_market_returns"]

    rng = random.Random(seed)
    assets = rng.sample(list(stock_rets.columns), k=min(n_assets, stock_rets.shape[1]))
    calc = BetaCalculator()

    betas_real: Dict[str, pd.Series] = {}
    betas_shrink: Dict[str, pd.Series] = {}
    betas_dcc: Dict[str, pd.Series] = {}

    for asset in assets:
        s = stock_rets[asset]
        m = market_rets

        # Realised beta
        betas_real[asset] = calc.realized_beta_vectorized(
            s.to_frame(asset), m, window=window
        )[asset]

        # Shrinkage beta
        betas_shrink[asset] = calc.beta_shrinkage_vectorized(
            s.to_frame(asset), m,
            short_window=window, annual_window=annual_window, prior_window=prior_window,
        )[asset]

        # DCC beta
        beta_dcc = calc.calculate_dcc_beta(s, m)
        if beta_dcc is None or beta_dcc.empty:
            beta_dcc = pd.Series(np.nan, index=s.index)
        betas_dcc[asset] = beta_dcc

    if show_plots:
        rows = int(np.ceil(n_assets / 2))
        cols = 2 if n_assets > 1 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharex=True)
        axes = np.atleast_1d(axes).flatten()
        for ax, asset in zip(axes, assets):
            betas_real[asset].plot(ax=ax, label="Realised", color="C0")
            betas_shrink[asset].plot(ax=ax, label="Shrinkage", color="C1", linestyle="--")
            betas_dcc[asset].plot(ax=ax, label="DCC", color="C2", linestyle=":")
            ax.set_title(f"Beta comparison – {asset}")
            ax.grid(True)
            ax.legend()
        fig.suptitle("Realised vs Shrinkage vs DCC betas", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        rows2 = int(np.ceil(n_assets / 2))
        cols2 = 2 if n_assets > 1 else 1
        fig2, axes2 = plt.subplots(rows2, cols2, figsize=(14, 4 * rows2), sharex=True)
        axes2 = np.atleast_1d(axes2).flatten()

        for ax2, asset in zip(axes2, assets):
            betas_real[asset].plot(ax=ax2, label="Realised", color="C0")
            betas_shrink[asset].plot(ax=ax2, label="Shrinkage", color="C1", linestyle="--")
            ax2.set_title(f"Realised vs Shrinkage – {asset}")
            ax2.grid(True)
            ax2.legend()

        fig2.suptitle("Realised beta vs Shrinkage beta (no DCC)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()



    return betas_real, betas_shrink, betas_dcc




# 1) Bêta réalisé vs. shrinkage  (5 actifs au hasard, fenêtre 90 j)
demo_realized_vs_shrinkage()

# 2) Filtre de liquidité cross-sectionnel  (fenêtre 20 j)
demo_liquidity_filter()

# 3) Comparaison Réalisé / Shrinkage / DCC  (4 actifs au hasard)
demo_compare_betas()

