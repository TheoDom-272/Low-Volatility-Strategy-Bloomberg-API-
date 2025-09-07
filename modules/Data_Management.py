import os
import pandas as pd
import datetime as dt
from pathlib import Path
import numpy as np
import math
import shutil



class DataAPI :
    """
    Manages data retrieval from different sources (Bloomberg, Yahoo Finance, or saved files)
    and provides a unified dictionary of market data.
    """

    def __init__(self,ticker,start_date,end_date,api_name = "Bloomberg") :
        """
        Initialize DataAPI with the desired data source and load the data immediately.

        :param ticker: List or string of index names to retrieve.
        :param start_date: Start date for data retrieval (YYYY-MM-DD).
        :param end_date: End date for data retrieval (YYYY-MM-DD).
        :param api_name: Which API to use ("Bloomberg", "Data Save").
        """

        self.api_name = api_name
        self.dict_data = self.load_data(ticker,start_date,end_date)



    def load_data(self,ticker,start_date,end_date):
        """Dispatch to the appropriate data-loading method based on api_name."""

        # If Bloomberg is requested, import and use the Bloomberg API wrapper
        if self.api_name == "Bloomberg":
            data_dict = self.main_bloomberg(ticker,start_date,end_date)

        # If "Data Save" is requested, load from locally saved files
        elif self.api_name == "Data Save":
            data_dict = self.main_datasave(ticker,start_date,end_date)

        return data_dict



    def main_bloomberg(self,tickers, start_date, end_date):
        """
        Retrieve index compositions and historical price/volume data from Bloomberg.

        Steps:
          1. Map universe names to Bloomberg tickers.
          2. For each quarter between start_date and end_date, fetch index membership.
          3. Save the evolving universe to Excel.
          4. Download historical PX_LAST, PX_VOLUME, and GICS sector fields.
          5. Compute daily returns for each stock and the index.
          6. Align stock returns with index returns and store in the output dict.
        """

        if self.api_name == "Bloomberg":
            # Instance of the BLP class used to fetch Bloomberg data
            from BloombergAPI import BLP
            blp = BLP()
        else:
            raise RuntimeError("Bloomberg API not available.")

        list_ticker = []


        index_mapping = {
            'S&P 500': 'SPX Index',
            'Russel 1000' : 'RIY Index',
            'Russel 2000': 'RTY Index',
            'Russel 3000': 'RAY Index',
        }

        if isinstance(tickers, str):
            tickers = index_mapping[tickers]
            list_ticker.append(tickers)
        else:
            for ticker in tickers:
                ticker_bloom = index_mapping[ticker]
                list_ticker.append(ticker_bloom)

        # Analysis period
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Instance of the BLP class used to fetch Bloomberg data
        blp = BLP()

        # Variables used to store retrieved data
        all_compositions = {}
        all_price_data = {}
        all_returns = {}
        data_dict = {}

        # Step 1: Import index compositions and stock prices
        print("Importing data...")

        # Loop over each selected market index
        for market_index_ticker in list_ticker:

            ticker_union = set()  # Set to collect all unique tickers
            composition_date_dict = {}  # Dictionary to store tickers by date

            # Create list of dates every years between start_date and end_date
            date_list = pd.date_range(start=start_date, end=end_date, freq='AS')

            for comp_date in date_list:
                comp_date_str = comp_date.strftime('%Y%m%d')
                print(f"Fetching composition for index {market_index_ticker} at date {comp_date_str}...")

                # Here we add the 'date' override parameter to the BDS call
                composition_data = blp.bds(
                    strSecurity=[market_index_ticker],
                    strFields=["INDX_MWEIGHT_HIST"],
                    strOverrideField = "END_DATE_OVERRIDE", strOverrideValue = comp_date_str
                )

                # Extract tickers for this date
                tickers_temp = []
                for key, df in composition_data.items():
                    tickers_temp.extend(df['Member'].tolist())

                # Remove duplicates for this date
                tickers_temp = list(set(tickers_temp))

                # Append " Equity" suffix for Bloomberg-compatible formatting
                tickers_temp = [ticker + " Equity" for ticker in tickers_temp]

                # Update the dictionary with this date and corresponding tickers
                composition_date_dict[comp_date_str] = tickers_temp

                # Add tickers to the global set
                ticker_union.update(tickers_temp)

            composition_by_year = {}
            for date_str, tickers_list in composition_date_dict.items():
                year = pd.to_datetime(date_str, format='%Y%m%d').year
                composition_by_year[year] = tickers_list

            composition_df = pd.Series(composition_by_year, name='Tickers').to_frame()
           
            df_lists = pd.DataFrame(composition_df['Tickers'].tolist(),index=composition_df.index) 
                             
            df_final = df_lists.T
            composition_df = (pd.DataFrame.from_dict(composition_by_year, orient="index").T) 
            tickers_in_index = list(ticker_union)  # Unique list of all tickers in the universe

            # --- Export annual composition to Excel ---
            output_dir = os.path.join(os.getcwd(), "Data")
            os.makedirs(output_dir, exist_ok=True)

            # Build the filename
            comp_path = os.path.join(
                output_dir,
                f"composition_by_year_{market_index_ticker.replace(' ', '_')}.xlsx"
            )

            # Export the DataFrame (index is year, column "Tickers")
            df_final.to_excel(comp_path, index=True, sheet_name="Composition")

            print(f"Composition by year exported to: {comp_path}")

            # Export universe tickers to an Excel file
            output_dir = os.path.join(os.getcwd(), "Data")
            tickers_df = pd.DataFrame(tickers_in_index, columns=["Tickers"])
            ticker_export_path = os.path.join(output_dir,
                                              f"tickers_univers_{market_index_ticker.replace(' ', '_')}.xlsx")
            tickers_df.to_excel(ticker_export_path, index=False)
            print(f"Universe tickers exported to: {ticker_export_path}")

            # Save the date → tickers mapping for this index
            all_compositions[market_index_ticker] = composition_date_dict

            # Fetch historical prices for index members
            print(f"Fetching historical prices for {len(tickers_in_index)} stocks in {market_index_ticker}...")
            price_data = blp.bdh(strSecurity=tickers_in_index, strFields=["PX_LAST", "PX_VOLUME", "GICS SECTOR"],
                                 startdate=start_date,
                                 enddate=end_date)
            all_price_data[market_index_ticker] = price_data

            gics_data = blp.bdp(strSecurity=tickers_in_index, strFields=["GICS_SECTOR_NAME"]) #Probleme ici

            # Retrieve historical prices, volumes, and sector codes
            df_prices = price_data["PX_LAST"]
            df_volumes = price_data["PX_VOLUME"]
            df_secteur = gics_data["GICS_SECTOR_NAME"] #A vérif

            # Compute returns for each stock and for the index
            stock_returns = {}
            stock_volumes = {}
            for field in price_data["PX_LAST"]:
                stock_returns[field] = price_data["PX_LAST"][
                    field].pct_change().dropna()
                stock_volumes[field] = price_data["PX_VOLUME"][field].dropna()

            market_data = blp.bdh(strSecurity=market_index_ticker, strFields=["PX_LAST"],
                                 startdate=start_date,
                                 enddate=end_date)

            # Retrieve index-level price series
            market_returns = market_data["PX_LAST"][
                market_index_ticker].pct_change().dropna()

            df_returns = pd.DataFrame(stock_returns)

            combined_returns = pd.concat([df_returns, market_returns], axis=1).dropna()
            aligned_stock_returns = combined_returns.drop(columns=[market_index_ticker])
            aligned_market_returns = combined_returns[[market_index_ticker]]

            # Save to final dictionary under expected keys
            data_dict[market_index_ticker] = {
                'prices': df_prices,
                'returns': combined_returns,
                'volumes': df_volumes,
                'sectors': df_secteur,
                'aligned_stock_returns': aligned_stock_returns,
                'aligned_market_returns': aligned_market_returns,
                'market_returns': market_returns,
                'composition': composition_df
            }


        # CSV export of all gathered data (one CSV par "feuille")
        output_dir = os.path.join(os.getcwd(), "Data")
        os.makedirs(output_dir, exist_ok=True)

        for index, content in data_dict.items():

            index_dir = os.path.join(output_dir, index.replace(' ', '_'))
            os.makedirs(index_dir, exist_ok=True)

            for key, df in content.items():

                if df is not None:
                    # Normaliser le nom de fichier (pas d'espaces, tout en minuscules)
                    filename = key.lower().replace(' ', '_') + '.csv'
                    path_csv = os.path.join(index_dir, filename)
                    df.to_csv(path_csv, index=True)

            print(f"Export CSV terminé pour {index} dans {index_dir}")

        return data_dict




    def compute_datasave(self,list_index, start_date, end_date):
        """
        Load pre-saved Excel files (Data Save mode) from the Data folder,
        filter by date, and reconstruct the same data structure as other methods.
        """


        data_dict = {}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(script_dir, "..", "Data"))
        print("Output dir:", output_dir)

        if not os.path.exists(output_dir):
            print("The 'Results' folder does not exist. No data to load.")
            return data_dict

        for index in list_index:
            print(f"Retrieving data for index {index}")
            file_path = os.path.join(output_dir, f"financial_data_{index}.xlsx")

            if os.path.exists(file_path):
                with pd.ExcelFile(file_path) as xls:
                    sheets = pd.read_excel(xls, sheet_name=None, index_col=0)
                    df_prices = sheets.get("Prices")
                    df_returns = sheets.get("Returns")
                    df_volumes = sheets.get("Volumes")
                    df_sectors = sheets.get("Sectors")
                    aligned_stock_returns = sheets.get("aligned stock returns")
                    aligned_market_returns = sheets.get("aligned market returns")
                    market_returns = sheets.get("market returns")

                # Convert index to datetime and filter date range
                if df_prices is not None:
                    df_prices.index = pd.to_datetime(df_prices.index)
                    df_prices = df_prices.loc[start_date:end_date]
                if df_returns is not None:
                    df_returns.index = pd.to_datetime(df_returns.index)
                    df_returns = df_returns.loc[start_date:end_date]
                if df_volumes is not None:
                    df_volumes.index = pd.to_datetime(df_volumes.index)
                    df_volumes = df_volumes.loc[start_date:end_date]
                if aligned_stock_returns is not None:
                    aligned_stock_returns.index = pd.to_datetime(aligned_stock_returns.index)
                    aligned_stock_returns = aligned_stock_returns.loc[start_date:end_date]
                if aligned_market_returns is not None:
                    aligned_market_returns.index = pd.to_datetime(aligned_market_returns.index)
                    aligned_market_returns = aligned_market_returns.loc[start_date:end_date]
                if market_returns is not None:
                    market_returns.index = pd.to_datetime(market_returns.index)
                    market_returns = market_returns.loc[start_date:end_date]

                data_dict[index] = {
                    "prices": df_prices,
                    "returns": df_returns,
                    "volumes": df_volumes,
                    "market returns": market_returns,
                    "combined returns": pd.concat([df_returns, market_returns], axis=1).dropna(),
                    "aligned_stock_returns": aligned_stock_returns,
                    "aligned_market_returns": aligned_market_returns,
                    "sectors": df_sectors
                }

        return data_dict


    def main_datasave(self,list_index, start_date, end_date):

        """
        Similar to compute_datasave but loads individual CSVs per sheet
        from a subfolder per index, then rebuilds the data_dict.
        """

        data_dict = {}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(script_dir, "..", "Data"))
        print("Output dir:", output_dir)

        if not os.path.exists(output_dir):
            print("The 'Results' folder does not exist. No data to load.")
            return data_dict

        for index in list_index:
            name_field = f"{index} Complet"
            index_dir = os.path.join(output_dir, name_field)
            if not os.path.isdir(index_dir):
                print(f"[Warning] Folder for {index} does not exist ({index_dir})")
                continue

            # Expected sheet names to load
            sheets = ["Prices", "Returns", "Volumes", "Sectors",
                      "aligned_stock_returns", "aligned_market_returns", "market_returns","composition_by_year"]

            # Load each CSV if available
            content = {}
            for sheet in sheets:
                
                if sheet =="composition_by_year":
                    csv_path = os.path.join(index_dir, f"{sheet}.xlsx")
                else:
                    csv_path = os.path.join(index_dir, f"{sheet}.csv")

                if os.path.exists(csv_path):
                    if sheet == "composition_by_year":
                        df = pd.read_excel(csv_path, index_col=0)
                    else:
                        df = pd.read_csv(
                            csv_path,
                            index_col=0,
                            parse_dates=True,
                            engine='python',
                        )

                        # Filter by date range and clean empty/zero columns
                        if sheet != "Sectors":
                            df = df.loc[start_date:end_date]
                            df.index = pd.to_datetime(df.index, errors='coerce')
                            df = df.dropna(how="all", axis=1)
                            df = df.loc[:, (df != 0).any(axis=0)]
                else:
                    df = None

                # Convert sheet names to consistent lowercase keys
                key = sheet.lower().replace(" ", "_")
                content[key] = df

            # Rebuild expected structure with correct key names
            data_dict[index] = {
                "prices": content["prices"],
                "returns": content["returns"],
                "volumes": content["volumes"],
                "sectors": content.get("sectors"),
                "aligned_stock_returns": content["aligned_stock_returns"],
                "aligned_market_returns": content["aligned_market_returns"],
                "market_returns": content["market_returns"],
                "composition": content["composition_by_year"]
            }

        return data_dict



class DataExportlast:
    """
    Provides methods to export portfolio compositions and history to Excel files.
    """

    def export_portfolio_compositionlast(self, portfolio_weights_df, rebal_date): # Surement à supprimer
        """
        Export a single rebalancing portfolio composition to an Excel file
        in the 'PORT' folder, with the filename including the rebalancing date.
        """

        # Create "PORT" folder if it doesn't exist
        port_folder = os.path.join(os.getcwd(), "PORT")
        if not os.path.exists(port_folder):
            os.makedirs(port_folder)

        # Format the filename with the rebalancing date
        if isinstance(rebal_date, (str,)):

            # Convert string to datetime if necessary
            rebal_date = pd.to_datetime(rebal_date)
        file_name = f"portfolio_{rebal_date.strftime('%Y%m%d')}.xlsx"
        file_path = os.path.join(port_folder, file_name)

        # Save DataFrame to Excel
        portfolio_weights_df.to_excel(file_path, index=True, header=["Weight"])
        print(f"Portfolio exported to: {file_path}")

    def export_portfolio_historylast(self, all_weights_df):
        """
        Export the full history of portfolio weights (Ticker, Weight, Date)
        to a single Excel file in the 'PORT' folder.
        """

        # Create "PORT" folder if it doesn't exist
        port_folder = os.path.join(os.getcwd(), "PORT")
        if not os.path.exists(port_folder):
            os.makedirs(port_folder)

        # Save the entire weight history to Excel
        file_path = os.path.join(port_folder, "API_bloom_portfolio_history.xlsx")
        all_weights_df.to_excel(file_path, index=False)
        print(f"API_bloom_portfolio_history exported to: {file_path}")

    def export_bench_historylast(self, all_weights_df):
        """
        Export the full history of portfolio weights (Ticker, Weight, Date)
        to a single Excel file in the 'PORT' folder.
        """

        # Create "PORT" folder if it doesn't exist
        port_folder = os.path.join(os.getcwd(), "PORT")
        if not os.path.exists(port_folder):
            os.makedirs(port_folder)

        # Save the entire weight history to Excel
        file_path = os.path.join(port_folder, "BENCH_portfolio_history.xlsx")
        all_weights_df.to_excel(file_path, index=False)
        print(f"BENCH_portfolio_history exported to: {file_path}")



class DataExport:
    """
    Export portfolio or benchmark weights to Excel files compatible with
    Bloomberg PORT.

    If the output exceeds Excel's row limit (1 000 000), the file is split
    into multiple chunks.  Each chunk keeps the same filename
    "<label>_portfolio_history.xlsx" but is saved in a numbered sub-folder
    (01/, 02/, …) under PORT/<label>/ so PORT can still discover them.
    """

    MAX_XLSX_ROWS = 100000         # Excel limit

    def _wide_to_long(self, weights_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a wide DataFrame (index = dates, columns = tickers) into a
        long table with columns Date ,Ticker, Weight.  Zero weights are
        dropped.  Uses the new pandas stack implementation (future_stack=True).
        """

        df = weights_df.copy()

        # Ensure index is datetime
        if not np.issubdtype(df.index.dtype, np.datetime64):
            df.index = pd.to_datetime(df.index, errors="coerce")


        df_long = (
            df.stack(future_stack=True)            # Series with MultiIndex (Date, Ticker)
            .rename_axis(index=["Date", "Ticker"])
            .reset_index(name="Weight")          # turn Series into DataFrame
        )

        # Keep only non-zero (and non-NaN) weights
        df_long = df_long[df_long["Weight"].notna() & (df_long["Weight"] != 0)]

        if df_long["Date"].isna().any():
            raise ValueError("Some dates could not be parsed to datetime.")
        
        df_long["Date"] = df_long["Date"].dt.strftime("%Y-%m-%d")

        return df_long

    def _ensure_port_folder(self) -> Path:
        """Return (or create) the ./PORT folder at project root."""
        port_folder = Path.cwd() / "PORT"
        port_folder.mkdir(exist_ok=True)
        return port_folder

    def _purge_label_folder(self, label: str) -> Path:
        """
        Delete ./PORT/<label>/ entirely (if it exists) and recreate it empty.
        Returns the freshly created Path object.
        """
        base = self._ensure_port_folder()
        label_dir = base / label
        if label_dir.exists():
            shutil.rmtree(label_dir)
        label_dir.mkdir(parents=True, exist_ok=True)
        return label_dir


    def export_history(self, weights_df: pd.DataFrame, label: str = "PORT") -> None:
        """
        Export the full weight history.

        • If ≤ 1 000 000 rows, one Excel file at ./PORT/<label>_portfolio_history.xlsx  
        • Otherwise, the DataFrame is split into N chunks, each saved as
          <label>/01/<label>_portfolio_history.xlsx,
          <label>/02/<label>_portfolio_history.xlsx, ...

        Before exporting, the existing ./PORT/<label>/ folder is purged so
        outdated inventory files are not re-uploaded accidentally.
        """
        df_long = self._wide_to_long(weights_df)
        n_rows  = len(df_long)

        # Simple case: single file
        if n_rows <= self.MAX_XLSX_ROWS:
            path = self._ensure_port_folder() / f"{label}_portfolio_history.xlsx"
            df_long.to_excel(path, index=False)
            print(f"File exported: {path.resolve()}")
            return

        # Large case: split into chunks
        label_root = self._purge_label_folder(label)     # start from a clean directory
        n_chunks   = math.ceil(n_rows / self.MAX_XLSX_ROWS)
        chunk_size = math.ceil(n_rows / n_chunks)

        for i in range(n_chunks):
            start, end = i * chunk_size, min((i + 1) * chunk_size, n_rows)
            df_chunk   = df_long.iloc[start:end]

            # Sub-folder numbered 01, 02, 03, …
            subdir = label_root / f"{i + 1:02d}"
            subdir.mkdir(exist_ok=True)

            file_path = subdir / f"{label}_portfolio_history.xlsx"
            df_chunk.to_excel(file_path, index=False)

            # Relative path for a cleaner console message
            rel = file_path.relative_to(label_root.parent)
            print(f"Chunk {i + 1}/{n_chunks} exported: {rel}")