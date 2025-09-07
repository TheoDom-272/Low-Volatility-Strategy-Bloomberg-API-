# Low Volatility Equity Strategy

> GUI  + Backtester  + Reporting  — Python project to research, backtest, and export a long‑only or long/short **low volatility** equity strategy on US universes (Russel 1000) with Bloomberg or cached data.


---

## Table of Contents
- [Quick Overview](#quick-overview)
- [Architecture & Modules](#architecture--modules)
- [Installation](#installation)
- [Launch the GUI](#launch-the-gui)
- [Data Sources: Bloomberg vs Data Save](#data-sources-bloomberg-vs-data-save)
- [Low Volatility Strategy — Methodology](#low-volatility-strategy--methodology)
  - [Universe & Calendar](#universe--calendar)
  - [Beta Measurement](#beta-measurement)
  - [Cross‑Sectional Liquidity Filter](#cross-sectional-liquidity-filter)
  - [Stock Selection](#stock-selection)
  - [Weight Optimization](#weight-optimization)
  - [Backtest & Costs](#backtest--costs)
  - [Dynamic Equal‑Weight Benchmark](#dynamic-equal-weight-benchmark)
- [Outputs & Reporting](#outputs--reporting)
- [Parameters (GUI)](#parameters-gui)
- [Disclaimer](#disclaimer)

---

## Quick Overview

- **Interface**: `tkinter` GUI to choose strategy, universe, frequencies, beta & optimization methods.
- **Backtester**: scheduled rebalancing + **daily weight drift** (buy‑and‑hold between rebalances), transaction costs, selection frequency decoupled from rebalancing frequency.
- **Betas**: Realized (rolling OLS), **Shrinkage** (Bayesian, fully vectorized with chunking), and an **experimental DCC** implementation (kept for research, not exposed in the GUI).
- **Liquidity filter**: **dollar‑volume** (price×volume) with **log1p + median/MAD** for robustness.
- **Optimization**:
  - **ERC** (Equal Risk Contribution) for Long and Short sleeves,
  - **Mean/Variance** (maximize empirical Sharpe).
- **Benchmark**: **equal‑weight**, rebuilt **each year** from index membership (if available).
- **Exports**: weights (Bloomberg PORT‑compatible) in `PORT/`, HTML report in `../Result/`, charts in `Graphique/`.

---

## Architecture & Modules

### `InvestmentStrategiesApp.py`
Main **GUI** app. Responsibilities:
- collects user inputs (API, strategy, beta method, universe, dates, frequencies, constraints, market‑neutral, costs, etc.)
- loads data via `Data_Management.DataAPI`.
- computes **betas** with `InvestmentStrategies.BetaCalculator`.
- runs the **Backtester** (`Backtester.Backtester`).
- exports weights via `Data_Management.DataExport`.
- builds the **report** with `PortfolioAnalysis.PortfolioStatistics`.

Highlights:
- **Automatic extension** of the requested `start_date` by **−900 days** to ensure enough history for rolling windows.
- **Market‑neutral** toggle shows/hides the *Short param* field.
- Dynamic universe list (Russel 1000/2000/3000) depending on the selected **API**.

### `PortfolioAnalysis.py` → `PortfolioStatistics`
Performance & risk analytics:
- Cumulative performance, drawdown, **rolling Sharpe** (annualized), **rolling volatility**, **rolling beta** vs benchmark.
- **Long/Short** sleeves shown separately if available.
- **Sector allocations** (avg.) for long / short / total and **relative exposure** vs benchmark.
- **Tracking error** ex‑post (annualized stdev of active returns) and **ex‑ante** (from asset covariance & average weights).
- **VaR 95%**: historical, parametric, and **Monte Carlo** (EWMA RiskMetrics + Cholesky).
- Generates an **HTML report** (in `../Result/Portfolio_Report.html`) and stores figures in `Graphique/`.

> **Windows**: opens the report through Edge using a hard‑coded path. Adjust if needed (see [Troubleshooting](#troubleshooting--notes)).

### `InvestmentStrategies.py`
Two key classes:

- **`BetaCalculator`**
  - `realized_beta_vectorized`: realized beta (≈ rolling OLS) computed **vectorially** via `numpy.sliding_window_view`.
  - `beta_shrinkage_vectorized`: **shrinkage** combining short‑term beta (estimator variance `V`) and long‑term (annual) beta (prior) with weight \(\tau^2 / (\tau^2 + V)\). Returns a time‑aligned DataFrame.
  - `beta_shrinkage_in_chunks`: splits the asset matrix into **chunks** to reduce memory footprint.
  - `calculate_dcc_beta` (experimental): attempts \(\beta_t = \frac{\mathrm{Cov}_t(r_i, r_m)}{\mathrm{Var}_t(r_m)}\) with univariate GARCH + DCC(1,1). Kept for **research** and demos.

- **`PortfolioManagement`**
  - `cross_sectional_liquidity_filter_with_history`: robust **log(dollar‑volume)** filter using **MAD z‑scores**; keeps strong positives, excludes names under a negative threshold (configurable).
  - `select_stocks`: picks **low‑β** for the **Long** sleeve and **high‑β** for the **Short** sleeve (if enabled), **widening** the `beta_threshold` until a **minimum number of stocks** is reached (percentage of available names).
  - `optimize_portfolio_erc`: **relative** ERC (balances risk contributions) for Long; Short uses positive intensities then flips sign.
  - `optimize_portfolio_mv`: maximizes empirical **Sharpe** over the window; equality constraints on weight sums.
  - `create_equal_weight_portfolio`: utility to produce an **equal‑weighted** benchmark.

### `Backtester.py` → `Backtester`
- **Calendar**: builds **rebalancing** dates (`Weekly=W‑FRI`, `Monthly=ME`, `Quarterly=QE`, `Annually=A`), then performs **universe selection** every `k` rebalances where \(k = \mathrm{round}(\text{Rebal}/\text{Select})\).
- **Daily drift**: between rebalances, weights follow \(w_t = w_{t-1}\,(1+r_t)\), then are **normalized separately** for Long/Short sleeves for tracking & PnL.
- **Costs**: applied **only** at rebalancing dates as `turnover × tc_bps`.
- **Benchmark**: **equal‑weight**, rebuilt from `composition_df` (if present), **re‑initialized each year**.
- **Outputs**: portfolio & benchmark return series, daily normalized weights, rebalancing weights, fees, and Long/Short sleeve returns (if applicable).

### `Data_Management.py`
- **`DataAPI`**
  - `main_bloomberg`:
    1) maps universes → Bloomberg tickers (`RIY Index`, `RTY Index`, `RAY Index`),
    2) fetches **annual compositions** via `BDS INDX_MWEIGHT_HIST` (with `END_DATE_OVERRIDE`),
    3) downloads **PX_LAST**/**PX_VOLUME**/**GICS** for all members,
    4) computes **returns** for assets + index,
    5) **aligns** stock & index returns for beta computations.
  - `main_datasave`: reloads pre‑saved **CSV/XLSX** from `Data/<Index Complet>/` (cached **Data Save** mode).
- **`DataExport`**
  - Exports weight histories in **long format** (Date/Ticker/Weight) for Bloomberg PORT.
  - **Splits** automatically into sub‑folders `PORT/<LABEL>/01/`, `02/`, … if more than **1,000,000** rows.

### `BloombergAPI.py` → `BLP`
Thin wrapper around `blpapi`:
- `bdh` (historical), `bdp` (reference), `bds` (lists — here, index membership with weights). **Requires** Bloomberg Desktop/Server and entitlements.

### `Etude.py`
**Demo** script (not used by the GUI):
- Realized vs Shrinkage beta (loop vs vectorized).
- Visualization of the **liquidity filter**.
- Overlay **Realized vs Shrinkage vs DCC** for a few names.

---

## Installation

> Python **3.10+** recommended.

```bash
# Create a virtual env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Core deps
pip install pandas numpy scipy matplotlib tqdm python-dateutil arch
pip install blpapi           # only if you use the Bloomberg mode
```

- **tkinter** ships with most Python distributions (otherwise install the system package for your OS).
- For **Bloomberg**, you need an **entitled** desktop and the Python SDK (`blpapi`).

---

## Launch the GUI

```bash
python InvestmentStrategiesApp.py
```

1) Pick **API**: **Data Save** (local cache) or **Bloomberg**.  
2) Choose **Low Volatility** strategy and **beta method** (Realized / Shrinkage).  
3) Select **universe** (Russel 1000 by default; 2000/3000 available with Bloomberg).  
4) Set dates, frequencies, constraints, costs, etc.  
5) Click **Run** → backtest, export, and HTML report.

---

## Data Sources: Bloomberg vs Data Save

- **Bloomberg** (`main_bloomberg`): builds the universe from **annual index memberships**, downloads prices/volumes + sectors, computes returns and aligned series (stocks vs index) for beta estimation.
- **Data Save** (`main_datasave`): reads ready‑made files from `Data/`:
  - expected folder: `Data/<Index Complet>/` (e.g. `Data/Russel 1000 Complet/`),
  - files: `Prices.csv`, `Returns.csv`, `Volumes.csv`, `Sectors.csv`, `aligned_stock_returns.csv`, `aligned_market_returns.csv`, `market_returns.csv`, `composition_by_year.xlsx`.

> Make sure headers (tickers) are consistent across prices/volumes/returns.

---

## Low Volatility Strategy — Methodology

### Universe & Calendar
- **Universes**: Russel 1000/2000/3000 (depending on API). The **annual** composition is used to constrain the investable set and to rebuild the **equal‑weight benchmark**.
- **Frequencies**:
  - **Selection Frequency** (e.g., quarterly): re‑select the investable set **every k rebalances**, where \(k = \mathrm{round}(\text{Rebal}/\text{Select})\).
  - **Rebalance Frequency** (e.g., monthly): optimization + transaction costs.

### Beta Measurement
- **Realized** (rolling window, default 90 days):
  \[ \beta_i = \frac{\mathrm{Cov}(r_i, r_m)}{\mathrm{Var}(r_m)} \]
  Fully **vectorized** for *M* assets × *T* dates.

- **Shrinkage**: combines **short‑term β** (90‑day window) and **long‑term β** (annual window) with Bayesian weights:
  \[
  \beta^{\text{shrunk}} = \frac{\tau^2}{\tau^2 + V}\,\beta^{\text{short}} + \frac{V}{\tau^2 + V}\,\beta^{\text{annual}}
  \]
  where \(V\) is the OLS estimator variance, and \(\tau^2\) is the empirical variance of \(\beta^{\text{short}}\) over `prior_window` (default 252 days).

- **DCC** (experimental): univariate GARCH + DCC(p,q) on standardized residuals to estimate time‑varying \(\text{Cov}_t\) and \(\text{Var}_t\) → \(\beta_t\). Kept for analysis, not available in the GUI.

> **Note**: the app **extends** the requested `start_date` by **900 days** to ensure enough history for beta windows.

### Cross‑Sectional Liquidity Filter
- Indicator: average **dollar‑volume** over the window (price × volume), transformed with `log1p`.
- Robust z‑scores: \(z = (x - \text{median}) / \text{MAD}\).
- Rule:
  - always keep assets with **high z‑scores** (very liquid),
  - among the remainder, **exclude** names below `volume_threshold` (in MAD units).

### Stock Selection
- Compute **mean β** over the **selection window** (e.g., 63 trading days for quarterly).
- **Long**: names with \(\overline{\beta}\) **below** the `beta_threshold` quantile (e.g., 50% → lower half).
- **Short** (if enabled): names **above** the \(1 - \text{beta_threshold}\) quantile.
- If fewer names than the **minimum** (`min_portfolio_stocks` as a % of available), **widen** `beta_threshold` by **+2.5 pp** steps until satisfied.

### Weight Optimization
Two approaches, **separately** for **Long** and **Short** sleeves:

- **ERC** (default): minimize dispersion of **relative risk contributions** \(RC = w \odot (\Sigma w)\). Constraints: \(w \ge 0\) for Long; Short uses **positive intensities** then multiplies by −1.

- **Mean/Variance**: maximize empirical **Sharpe** over the window with equality constraints on total weights.

**Target exposure**:
- If **Market Neutral**: Long = Short = `short_allocation`.
- Else: Long = `1 + short_allocation`, Short = `short_allocation`.

### Backtest & Costs
- At each **rebalancing**: (re)optimize weights; apply **costs** on **turnover** × `tc_bps`.
- Between rebalances: **daily weight drift** (buy‑and‑hold) with separate Long/Short normalization for tracking & returns.

### Dynamic Equal‑Weight Benchmark
- **Rebuilt annually** from **index members** (file `composition_by_year.xlsx` or via Bloomberg BDS), equally weighted across the active names that year.

---

## Outputs & Reporting

- **Excel exports** (Bloomberg **PORT**‑compatible):
  - `PORT/PORT_portfolio_history.xlsx` (or sub‑folders `PORT/PORT/01/…` for large datasets),
  - columns: `Date,Ticker,Weight` (dates **YYYY‑MM‑DD**).
- **Charts** in `Graphique/`: cumulative performance, drawdown, rolling Sharpe, rolling vol, rolling beta, costs, sector allocations, relative exposure.
- **HTML report**: `../Result/Portfolio_Report.html` (opens a browser; Edge by default on Windows in the provided code).

---

## Parameters (GUI)

| Parameter | Description | Values / Default |
|---|---|---|
| **API Choice** | Data source | `Bloomberg` \| `Data Save` (default) |
| **Select Strategy** | Strategy | `Low Volatility` |
| **Beta Method** | Beta methodology | `Beta Realized` \| `Beta Shrinkage` |
| **Universe** | Indices | `Russel 1000` (+ `2000`, `3000` with Bloomberg) |
| **Start / End Date** | Study period | `2020‑01‑03` → `2024‑12‑31` (default) |
| **Selection Frequency** | Universe selection frequency | `Weekly`/`Monthly`/`Quarterly`/`Annually` |
| **Rebalance Frequency** | Rebalancing frequency | same |
| **Minimum portfolio stocks (%)** | **Minimum** long basket size (% of available names) | `15` |
| **Liquidity Threshold (MAD units)** | Exclusion threshold | `-1` (conservative) |
| **Beta Threshold** | Initial low‑β quantile | `0.50` |
| **Short param** | Short exposure (see Market Neutral) | `1` |
| **Transaction Cost (bp)** | Cost per unit of turnover | `10` |
| **Market Neutral** | Long = Short if checked | unchecked by default |
| **Optimisation Method** | Optimization method | `ERC` (default) or `Mean/Variance` |

> Effective backtest start will be no earlier than after the **−900‑day** extension used to build beta windows.


```

## Disclaimer

This repository is intended for **quant research & engineering**. Backtest results are **hypothetical** and do **not** guarantee future performance. This is **not** investment advice.
