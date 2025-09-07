
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import pandas as pd



from InvestmentStrategies import BetaCalculator,PortfolioManagement
from Data_Management import DataAPI,DataExport
from Backtester import Backtester
from PortfolioAnalysis import PortfolioStatistics




class InvestmentStrategiesApp(tk.Tk):
    """
    Main application class that manages the GUI window and orchestrates
    configuration, execution, and reporting of investment strategies via Tkinter.
    """


    def __init__(self):
        """
        Initialize the Tkinter window, set default configuration variables,
        and build the user interface widgets.
        """

        super().__init__()

        self.title("Investment Strategies")
        self.geometry("800x600")

        # Initialization of variables with default values at application launch
        self.strategy = tk.StringVar(value="Low Volatility")
        self.api_choice = tk.StringVar(value="Data Save")
        self.beta_method = tk.StringVar(value="Beta Shrinkage")
        self.universe_selection = []
        self.backtest_enabled = tk.BooleanVar(value=True)
        self.start_date = tk.StringVar(value="2020-01-03")
        self.end_date = tk.StringVar(value="2024-12-31")
        self.rebalance_freq = tk.StringVar(value="Monthly")
        self.selection_freq = tk.StringVar(value="Quarterly")
        self.min_portfolio_stocks = tk.StringVar(value="15")
        self.liquidity_threshold = tk.StringVar(value="-1")
        self.method_optim = tk.StringVar(value="ERC")
        self.beta_threshold = tk.StringVar(value="0.50")
        self.short_allocation = tk.StringVar(value="1")
        self.tc_bps = tk.StringVar(value="10")  # transaction costs in basis points
        self.market_neutral = tk.BooleanVar(value=False)
        self.market_neutral.trace_add("write", self.toggle_short_param)
        self.universe_vars: dict[str, tk.BooleanVar] = {}   # {universe: tk.BooleanVar}
        self.universe_checks: dict[str, tk.Checkbutton] = {}  # {universe: widget}

        # Build the user interface
        self.create_widgets()

    def create_widgets(self):

        """
       Construct and arrange all GUI components, including sections for:
         - API selection
         - Strategy selection
         - Universe selection
         - Management parameters
         - Backtest configuration
         - Run button
        """


        # Main frame for all interface sections
        main_frame = tk.Frame(self)
        main_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # API section
        api_frame = tk.LabelFrame(main_frame, text="API", padx=10, pady=10)
        api_frame.grid(row=0, column=0, sticky="w", padx=5, pady=5)

        # Strategies section
        strategies_frame = tk.LabelFrame(main_frame, text="Strategies", padx=10, pady=10)
        strategies_frame.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        # Dropdown for API choice
        api_label = tk.Label(api_frame, text="API Choice:")
        api_label.grid(row=0, column=0, sticky="w", padx=5)
        api_menu = ttk.Combobox(api_frame, textvariable=self.api_choice,
                                     values=["Bloomberg","Data Save"])
        api_menu.grid(row=0, column=1, padx=5)

        # Dropdown for strategy selection
        strategy_label = tk.Label(strategies_frame, text="Select Strategy:")
        strategy_label.grid(row=0, column=0, sticky="w", padx=5)
        strategy_menu = ttk.Combobox(strategies_frame, textvariable=self.strategy,
                                     values=["Low Volatility"])
        strategy_menu.grid(row=0, column=1, padx=5)

        # Bind the strategy change event to update the beta calculation section
        strategy_menu.bind("<<ComboboxSelected>>", self.update_beta_section)

        # Management parameters section
        management_frame = tk.LabelFrame(main_frame, text="Management Parameters", padx=10, pady=10)
        management_frame.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        # Minimum portfolio stocks (%)
        min_stocks_label = tk.Label(management_frame, text="Minimum portfolio stocks (%):")
        min_stocks_label.grid(row=0, column=0, sticky="w", padx=5)
        min_stocks_entry = tk.Entry(management_frame, textvariable=self.min_portfolio_stocks)
        min_stocks_entry.grid(row=0, column=1, padx=5)

        # Liquidity Threshold
        liquidity_label = tk.Label(management_frame, text="Liquidity Threshold (in unit of std):")
        liquidity_label.grid(row=1, column=0, sticky="w", padx=5)
        liquidity_entry = tk.Entry(management_frame, textvariable=self.liquidity_threshold)
        liquidity_entry.grid(row=1, column=1, padx=5)

        # Beta Threshold
        liquidity_label = tk.Label(management_frame, text="Beta Threshold:")
        liquidity_label.grid(row=2, column=0, sticky="w", padx=5)
        liquidity_entry = tk.Entry(management_frame, textvariable=self.beta_threshold)
        liquidity_entry.grid(row=2, column=1, padx=5)

        # Short weights
        self.short_label = tk.Label(management_frame, text="Short param:")
        self.short_label.grid(row=3, column=0, sticky="w", padx=5)
        self.short_entry = tk.Entry(management_frame, textvariable=self.short_allocation)
        self.short_entry.grid(row=3, column=1, padx=5)

        #Transactions fees
        tk.Label(management_frame, text="Transaction Cost (bp):").grid(row=4, column=0, sticky="w", padx=5)
        tk.Entry(management_frame, textvariable=self.tc_bps).grid(row=4, column=1, padx=5)

        # Market-neutral checkbox
        neutral_chk = tk.Checkbutton(management_frame,text="Market Neutral (Long = Short)",variable=self.market_neutral)
        neutral_chk.grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=(5,0))

        self.toggle_short_param()

        # Optimization Method
        optim_label = tk.Label(management_frame, text="Optimisation Method:")
        optim_label.grid(row=5, column=0, sticky="w", padx=5)
        optim_menu = ttk.Combobox(management_frame, textvariable=self.method_optim,
                                  values=["ERC", "Mean/Variance"])
        optim_menu.grid(row=5, column=1, padx=5)


        # Section "Beta Calculation Method"
        self.beta_method_frame = tk.LabelFrame(main_frame, text="Beta Calculation Methodology", padx=10, pady=10)
        self.beta_method_label = tk.Label(self.beta_method_frame, text="Select Beta Method:")
        self.beta_method_menu = ttk.Combobox(self.beta_method_frame, textvariable=self.beta_method,
                                             values=["Beta Realized", "Beta Shrinkage"])

        # Universe section
        universe_frame = tk.LabelFrame(main_frame, text="Universe", padx=10, pady=10)
        universe_frame.grid(row=2, column=0, sticky="w", padx=5, pady=5)



        self.universe_frame = tk.LabelFrame(main_frame, text="Universe", padx=10, pady=10)
        self.universe_frame.grid(row=2, column=0, sticky="w", padx=5, pady=5)

        tk.Label(self.universe_frame, text="Select Universes:").grid(row=0, column=0, sticky="w", padx=5)

        # Build initial list
        self.refresh_universe_options()

        # Rebuild every time the API combobox changes
        self.api_choice.trace_add("write", self.refresh_universe_options)

        # Backtester section
        # Backtest parameter inputs (dates and frequencies)
        backtest_frame = tk.LabelFrame(main_frame, text="Backtester", padx=10, pady=10)
        backtest_frame.grid(row=3, column=0, sticky="w", padx=5, pady=5)

        params_frame = tk.Frame(backtest_frame)
        params_frame.grid(row=0, column=0, sticky="w", padx=10)

        tk.Label(params_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, sticky="w")
        tk.Entry(params_frame, textvariable=self.start_date).grid(row=0, column=1, padx=5)
        tk.Label(params_frame, text="End Date (YYYY-MM-DD):").grid(row=1, column=0, sticky="w")
        tk.Entry(params_frame, textvariable=self.end_date).grid(row=1, column=1, padx=5)
        tk.Label(params_frame, text="Selection Frequency:").grid(row=2, column=0, sticky="w")
        ttk.Combobox(params_frame, textvariable=self.selection_freq,
                     values=["Weekly", "Monthly", "Quarterly", "Annually"]).grid(row=2, column=1, padx=5)
        tk.Label(params_frame, text="Rebalance Frequency:").grid(row=3, column=0, sticky="w")
        ttk.Combobox(params_frame, textvariable=self.rebalance_freq,
                     values=["Weekly", "Monthly", "Quarterly", "Annually"]).grid(row=3, column=1, padx=5)

        # Run button"
        run_button = tk.Button(main_frame, text="Run", command=self.run_strategy)
        run_button.grid(row=5, column=0, pady=20)

        # Initial update to show/hide beta calculation section based on initial strategy selection
        self.update_beta_section()


    def toggle_short_param(self, *args):
        """Masque le champ 'Short param' si market_neutral est coché."""
        if self.market_neutral.get():
            self.short_label.grid_remove()
            self.short_entry.grid_remove()
        else:
            self.short_label.grid()
            self.short_entry.grid()
    
    def refresh_universe_options(self, *args):
        """
        Rebuild the universe check-boxes every time the API choice changes.
        Data Save  = only Russell 1000
        Bloomberg  = Russell 1000, Russell 2000, Russell 3000
        """

        # Clear previous widgets
        for chk in self.universe_checks.values():
            chk.destroy()
        self.universe_vars.clear()
        self.universe_checks.clear()

        # Determine the allowed universes
        if self.api_choice.get() == "Data Save":
            allowed = ["Russel 1000"]
        else:  # Bloomberg
            allowed = ["Russel 1000", "Russel 2000", "Russel 3000"]

        # Re-create check-boxes
        for i, name in enumerate(allowed):
            var = tk.BooleanVar()
            chk = tk.Checkbutton(self.universe_frame,
                                text=name,
                                variable=var,
                                command=self.update_universe)  # keeps self.universe_selection in sync
            chk.grid(row=i, column=1, sticky="w", padx=5)
            self.universe_vars[name] = var
            self.universe_checks[name] = chk
        
        if "Russel 1000" in allowed:
            self.universe_vars["Russel 1000"].set(True)
            self.update_universe()

    def update_universe(self):

        """
        Update the list of selected investment universes based on the state
        of the S&P 500 and Russell 1000.
        """

        self.universe_selection = [u for u, var in self.universe_vars.items() if var.get()]

    def toggle_backtest_parameters(self):
        """
        Show or hide the backtest date and frequency input fields depending
        on whether the 'Enable Backtest' checkbox is checked.
        """

        if self.backtest_enabled.get():
            self.backtest_parameters_frame.grid()
        else:
            self.backtest_parameters_frame.grid_forget()

    def update_beta_section(self, event=None):
        """
        Show or hide the beta calculation method section based on the
        currently selected strategy.
        """

        strategy = self.strategy.get()
        if strategy == "Low Volatility":
            self.beta_method_frame.grid(row=1, column=0, sticky="w", padx=5, pady=5)
            self.beta_method_label.grid(row=0, column=0, sticky="w", padx=5)
            self.beta_method_menu.grid(row=0, column=1, padx=5)
        else:
            self.beta_method_frame.grid_forget()

    def run_strategy(self):

        """
        Retrieve user inputs, load data via DataAPI, and dispatch execution
        to either a one-time calculation or a full backtest, logging progress.
        """

        api = self.api_choice.get()
        strategy = self.strategy.get()
        beta_method = self.beta_method.get()
        universes = self.universe_selection
        backtest_enabled = self.backtest_enabled.get()
        min_stocks = float(self.min_portfolio_stocks.get()) / 100.0
        liq_threshold = float(self.liquidity_threshold.get())
        method_optim = str(self.method_optim.get())
        beta_threshold = float(self.beta_threshold.get())
        short_allocation = float(self.short_allocation.get())
        tc_bps = float(self.tc_bps.get())
        market_neutral = self.market_neutral.get()

        #if backtest_enabled:

        # Retrieve start and end dates for backtest
        user_start_date = self.start_date.get() # Backtest start date
        initial_start_date = user_start_date # Preserve original date
        start_dt_user = datetime.strptime(user_start_date, "%Y-%m-%d")
        extended_start_dt = start_dt_user - timedelta(days=900) # Fetch extra 900 days for rolling beta
        start_date = extended_start_dt.strftime("%Y-%m-%d")

        end_date = self.end_date.get()


        # Check if the period between dates is sufficient
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        min_days = 365 # Minimum 1 year for backtest
        delta_days = (end_dt - start_dt).days
        if delta_days < min_days:
            start_dt = end_dt - timedelta(days=min_days) - timedelta(days=900) # Ensure at least 900 days plus rolling window margin
            self.start_date.set(start_dt.strftime("%Y-%m-%d"))


        rebalance_freq = self.rebalance_freq.get()
        selection_freq = self.selection_freq.get()
        print(f"Running backtest from {start_dt_user} to {end_date} with {selection_freq} selection frequency and {rebalance_freq} rebalance frequency.")

        print(f"Selected strategy: {strategy}")
        print(f"Beta calculation method: {beta_method}")
        print(f"Selected universes: {universes}")


        Data_api = DataAPI(universes, start_date, end_date, api_name=api)
        data_dict = Data_api.dict_data


        # Execute Low Volatility strategy if selected
        if strategy == "Low Volatility":
            print("run low vol strat")
            portfolio_df = self.run_Low_volatility_strategie(data_dict,beta_method,backtest_enabled,
                                                             initial_start_date,end_date,rebalance_freq,
                                                             selection_freq,min_stocks,liq_threshold,beta_threshold,method_optim,short_allocation,tc_bps,market_neutral)



    def run_Low_volatility_strategie(self,data_dict,beta_method,backtest_enabled,start_date,end_date,
                                     rebalance_freq,selection_freq,min_stocks,liq_threshold,beta_threshold,method_optim,short_allocation,tc_bps,market_neutral):
        """
        Execute the Low Volatility strategy workflow:
          1. Gather price, return, volume, and sector data.
          2. Compute betas using the selected method.
          3. Concatenate all dataframes and align columns.
          4. If backtest is disabled, perform stock selection and single-date optimization.
          5. If backtest is enabled, run the full backtester and export results to Excel.
          6. Generate performance report and plots via PortfolioStatistics.
        """

        all_prices = []
        all_returns = []
        all_volumes = []
        all_secteurs=[]
        all_composition=[]

        calculator = BetaCalculator()
        beta_df_dict = {}

        # For each index in the investment universe, calculate betas
        for index, attribute in data_dict.items():

            # Retrieve price DataFrame
            prices_df = attribute.get("prices")
            if prices_df is not None: all_prices.append(prices_df)

            # Retrieve return DataFrame
            returns_df = attribute.get("returns")
            if returns_df is not None: all_returns.append(returns_df)

            # Retrieve volume DataFrame
            volumes_df = attribute.get("volumes")
            if volumes_df is not None: all_volumes.append(volumes_df)

            # Retrieve sector DataFrame
            secteur_df = attribute.get("sectors")
            if secteur_df is not None: all_secteurs.append(secteur_df)

            # Retrieve sector DataFrame
            composition_df = attribute.get("composition")
            if composition_df is not None: all_composition.append(composition_df)

            # Retrieve aligned stock and market returns
            aligned_stock_returns = attribute.get("aligned_stock_returns")
            aligned_market_returns = attribute.get("aligned_market_returns")

            # Begin beta calculations
            print(f"Beginning of beta calculations for the stocks of the {index}")
            if beta_method == "Beta Realized":
                betas_df = calculator.realized_beta_vectorized(aligned_stock_returns, aligned_market_returns)
                beta_df_dict[index] = betas_df
            elif beta_method == "Beta Shrinkage":
                betas_df = calculator.beta_shrinkage_in_chunks(
                    aligned_stock_returns,
                    aligned_market_returns,
                    short_window=90,  # adapte si besoin
                    annual_window=252,
                    prior_window=252,
                    n_chunks=8  # plus de chunks = moins de RAM par chunk
                )
                beta_df_dict[index] = betas_df
                # betas_df = calculator.beta_shrinkage_vectorized(aligned_stock_returns, aligned_market_returns)
                # beta_df_dict[index] = betas_df
            elif beta_method == "DCC Beta":
                for stock in aligned_stock_returns.columns:
                    print(stock)
                    stock_returns_df = aligned_stock_returns[stock]
                    dynamic_beta = calculator.calculate_dcc_beta(stock_returns_df, aligned_market_returns)

                    beta_df_dict[stock] = dynamic_beta



        all_prices_df = pd.concat(all_prices, axis=1, join="outer") if all_prices else pd.DataFrame()
        all_returns_df = pd.concat(all_returns, axis=1, join="outer") if all_returns else pd.DataFrame()
        all_volumes_df = pd.concat(all_volumes, axis=1, join="outer") if all_volumes else pd.DataFrame()
        all_betas_df = pd.concat(beta_df_dict, axis=1)
        all_secteurs_df = pd.concat(all_secteurs, axis=1, join="outer") if all_secteurs else pd.DataFrame()
        all_composition_df = pd.concat(all_composition,axis=1, join="outer") if all_composition else pd.DataFrame()


        all_prices_df   = self._clean_index(all_prices_df)
        all_returns_df  = self._clean_index(all_returns_df)
        all_volumes_df  = self._clean_index(all_volumes_df)
        all_betas_df    = self._clean_index(all_betas_df)
        all_composition_df = self._clean_index(all_composition_df)



        if beta_method == "DCC Beta":
            all_betas_df.columns = all_betas_df.columns.get_level_values(0)
        else:
            all_betas_df.columns = all_betas_df.columns.get_level_values(1)

        Backtest_class = Backtester(all_betas_df, all_prices_df, all_returns_df, all_volumes_df,start_date,method_optim,short_allocation, rebalance_freq,
                                    selection_freq,min_stock=min_stocks,volume_threshold=liq_threshold,beta_threshold=beta_threshold,tc_bps= tc_bps,composition_df=all_composition_df,market_neutral=market_neutral)

        portfolio_df,allocation_history,rebal_weights_df,weights_eq, port_returns_eq,fees_series,port_long_returns, port_short_returns =Backtest_class.Backtest()

        print("Backtest completed.")
        print("Exporting to Excel...")
        exporter = DataExport()
        exporter.export_history(rebal_weights_df, label="PORT")
        #exporter.export_history(weights_eq, label="BENCH")

        print("HTML report generation...")
        #launch the HTML
        portfolio_series = portfolio_df
        benchmark_series = port_returns_eq

        analyzer = PortfolioStatistics(
            portfolio_returns=portfolio_series,
            benchmark_returns=benchmark_series,
            sectors_df=all_secteurs_df,
            fees_series=fees_series,
            weights_df=rebal_weights_df,  # your rebalancing weights over time
            bench_weights_df=weights_eq,  # benchmark (equal‐weight) weights
            port_long_returns=port_long_returns,
            port_short_returns=port_short_returns,
            asset_returns_df=all_returns_df,
            all_returns_df=all_returns_df
        )

        analyzer.calculate_statistics()
        analyzer.generate_plots()
        report_path = analyzer.generate_html_report()

        print("end")
        return portfolio_df
    
    def _clean_index(self,df):
        """
        Ensure the DataFrame index is a tz-naive, unique, sorted DatetimeIndex.
        """

        if df.empty:
            return df                             

        # Force DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce")

        # Make index tz-naive
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        # Drop duplicates (keep first occurrence)
        df = df.loc[~df.index.duplicated()]

        # Sort in ascending order
        df = df.sort_index()

        return df



# Launch the Tkinter application
if __name__ == "__main__":
    app = InvestmentStrategiesApp()
    app.mainloop()
