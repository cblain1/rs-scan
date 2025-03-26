import datetime as dt
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def analyze_all_periods() -> None:
    """
    Analyze and save rankings for all lookback periods (5, 20, 50, 100, 200 day).
    """
    periods = {
        "5 Day": "5day_leaders_laggards.csv",
        "20 Day": "20day_leaders_laggards.csv",
        "50 Day": "50day_leaders_laggards.csv",
        "100 Day": "100day_leaders_laggards.csv",
        "200 Day": "200day_leaders_laggards.csv"
    }
    
    for period, output_file in periods.items():
        logging.info(f"\nAnalyzing {period} relative strength...")
        
        # Load and analyze data
        df = pd.read_csv("relative_strength_results.csv")
        df[period] = pd.to_numeric(df[period], errors="coerce")
        
        # Calculate deciles
        df["Decile"] = pd.qcut(df[period], q=10, labels=False)
        
        # Get top and bottom deciles
        leaders = df[df["Decile"] == 9].copy()
        laggards = df[df["Decile"] == 0].copy()
        
        # Add labels
        leaders["Group"] = "Leaders"
        laggards["Group"] = "Laggards"
        
        # Calculate averages
        leader_avg = leaders[period].mean()
        laggard_avg = laggards[period].mean()
        
        # Combine and sort
        result = pd.concat([leaders, laggards])
        result = result.sort_values(by=period, ascending=False)
        result = result.drop("Decile", axis=1)
        
        # Add summary rows
        summary_rows = pd.DataFrame([
            {"Symbol": "LEADERS_AVG", "Group": "Summary", period: leader_avg},
            {"Symbol": "LAGGARDS_AVG", "Group": "Summary", period: laggard_avg}
        ])
        
        # Add empty values for other columns in summary
        for col in result.columns:
            if col not in ["Symbol", "Group", period]:
                summary_rows[col] = ""
        
        # Combine and save
        final_result = pd.concat([result, summary_rows])
        final_result.to_csv(output_file, index=False)
        
        # Log summary
        print(f"\n{period} Analysis:")
        print(f"Leaders: {len(leaders)} stocks, Average: {leader_avg:.2f}%")
        print(f"Laggards: {len(laggards)} stocks, Average: {laggard_avg:.2f}%")
        print(f"Results saved to {output_file}")


def analyze_rs_deciles(return_column: str) -> None:
    """
    Analyze relative strength results by deciles and save leaders/laggards.

    Args:
        return_column (str): Column name to analyze ('5 Day', '20 Day', '50 Day', '100 Day', '200 Day')
    """
    # Load the RS results
    df = pd.read_csv("relative_strength_results.csv")

    # Convert any 'N/A' values to NaN for proper numerical analysis
    df[return_column] = pd.to_numeric(df[return_column], errors="coerce")

    # Calculate deciles
    df["Decile"] = pd.qcut(df[return_column], q=10, labels=False)

    # Get top and bottom deciles
    leaders = df[df["Decile"] == 9].copy()  # Top decile
    laggards = df[df["Decile"] == 0].copy()  # Bottom decile

    # Add a label column
    leaders["Group"] = "Leaders"
    laggards["Group"] = "Laggards"

    # Calculate average returns for each group
    leader_avg = leaders[return_column].mean()
    laggard_avg = laggards[return_column].mean()

    # Combine and sort by the return column
    result = pd.concat([leaders, laggards])
    result = result.sort_values(by=return_column, ascending=False)

    # Drop the Decile column
    result = result.drop("Decile", axis=1)

    # Create summary rows
    summary_rows = pd.DataFrame(
        [
            {"Symbol": "LEADERS_AVG", "Group": "Summary", return_column: leader_avg},
            {"Symbol": "LAGGARDS_AVG", "Group": "Summary", return_column: laggard_avg},
        ]
    )

    # Add empty values for other columns in summary rows
    for col in result.columns:
        if col not in ["Symbol", "Group", return_column]:
            summary_rows[col] = ""

    # Combine results with summary
    final_result = pd.concat([result, summary_rows])

    # Save to CSV
    final_result.to_csv("rs_leader_laggard.csv", index=False)

    # Log summary
    print(f"\nAnalysis based on {return_column} returns:")
    print(f"Number of leaders: {len(leaders)}")
    print(f"Number of laggards: {len(laggards)}")
    print(f"Average leader return: {leader_avg:.2f}%")
    print(f"Average laggard return: {laggard_avg:.2f}%")
    print("Results saved to rs_leader_laggard.csv")

    # To see the counts in each decile:
    print("\nDecile Counts:\n", df["Decile"].value_counts().sort_index())


def calculate_advanced_relative_strength(
    index_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    lookback_periods: list = [5, 20, 50, 100, 200],
) -> dict:
    """
    Calculate advanced Relative Strength analysis with multiple lookback periods.

    Args:
        index_df (pd.DataFrame): DataFrame containing index price data
        stock_df (pd.DataFrame): DataFrame containing stock price data
        lookback_periods (list): List of periods to calculate relative strength

    Returns:
        dict: Dictionary of relative strength metrics across different periods
    """
    if len(index_df) < max(lookback_periods) or len(stock_df) < max(lookback_periods):
        logging.warning(
            "Insufficient data for comprehensive Relative Strength analysis."
        )
        return {period: np.nan for period in lookback_periods}

    results = {}

    for period in lookback_periods:
        # Calculate rolling returns
        index_rolling_return = index_df["Close"].pct_change(period)
        stock_rolling_return = stock_df["Close"].pct_change(period)

        # Relative strength as the difference in rolling returns
        relative_strength = (stock_rolling_return - index_rolling_return).iloc[-1] * 100

        results[period] = round(relative_strength, 2)

    # Additional metrics
    results["latest_price_ratio"] = (
        stock_df["Close"].iloc[-1] / index_df["Close"].iloc[-1]
    )

    return results


def analyze_relative_strength(
    index_df: pd.DataFrame, stock_df: pd.DataFrame, symbol: str, results_list: list
) -> None:
    """
    Analyze relative strength and append results to a list for CSV export.

    Args:
        index_df (pd.DataFrame): DataFrame containing index price data
        stock_df (pd.DataFrame): DataFrame containing stock price data
        symbol (str): Stock symbol
        results_list (list): List to store results for CSV export
    """
    rs_metrics = calculate_advanced_relative_strength(index_df, stock_df)

    # Create a row for CSV: symbol, 5D, 20D, 50D, 100D, 200D, Ratio
    row = [
        symbol,
        rs_metrics.get(5, "N/A"),
        rs_metrics.get(20, "N/A"),
        rs_metrics.get(50, "N/A"),
        rs_metrics.get(100, "N/A"),
        rs_metrics.get(200, "N/A"),
        round(rs_metrics.get("latest_price_ratio", "N/A"), 4)
        if rs_metrics.get("latest_price_ratio") not in ["N/A", None]
        else "N/A",
    ]
    results_list.append(row)


def save_price_data(
    price_data: pd.DataFrame, symbol: str, output_folder: str = "output/price_data"
):
    """Save price data to a CSV file."""
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"{symbol}_price_data.csv")
    price_data.to_csv(filename, index=True)
    print(f"Price data saved to: {filename}")


def load_price_data(
    symbol: str, price_data_folder: str = "output/price_data"
) -> Optional[pd.DataFrame]:
    """
    Load price data from saved CSV files.

    Args:
        symbol: Stock symbol
        price_data_folder: Folder containing price data files

    Returns:
        DataFrame with price data or None if file not found
    """
    try:
        file_path = Path(price_data_folder) / f"{symbol}_price_data.csv"
        if not file_path.exists():
            logging.warning(f"Price data file not found for {symbol}")
            return None

        df = pd.read_csv(file_path)

        return df

    except Exception as e:
        logging.error(f"Error loading price data for {symbol}: {e}")


def download_or_load_price_data(
    stock: str, start_date: dt.datetime, end_date: dt.datetime, live: bool
) -> Optional[pd.DataFrame]:
    """Download or load price data for a single stock.

    Args:
        stock: Stock symbol
        start_date: Start date for data
        end_date: End date for data
        live: Whether to download fresh data

    Returns:
        Clean DataFrame or None if data unavailable
    """
    try:
        if live:
            price_dataframe = yf.download(stock, start=start_date, end=end_date)
            clean_dataframe = clean_price_data(price_dataframe)
            save_price_data(clean_dataframe, stock)
            time.sleep(0.1)
            return clean_dataframe
        else:
            price_dataframe = load_price_data(stock)
            if price_dataframe is not None:
                return clean_price_data(price_dataframe)
            return None

    except Exception as e:
        logging.error(f"Error downloading/loading price data for {stock}: {e}")
        return None


def clean_price_data(df):
    """Clean price data DataFrame. Sets Date column as index

    Args:
        df: DataFrame with price data

    Returns:
        pd.DataFrame: Cleaned DataFrame with Date as index
    """
    # Create a fresh copy to avoid chained indexing
    df = df.copy()

    # Reset index to make 'index' a column
    df = df.reset_index()

    # Drop unwanted columns safely
    columns_to_drop = ["index", "Unnamed: 0"]
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns)

    # Handle multi-level columns
    if isinstance(df.columns, pd.MultiIndex):
        if "Ticker" in df.columns.names:
            df.columns = df.columns.droplevel("Ticker")

    # Filter for desired columns
    desired_columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
    available_columns = [col for col in desired_columns if col in df.columns]
    df = df[available_columns].copy()

    # Convert Date to datetime and set as index safely
    if "Date" in df.columns:
        # Use loc for assignment
        df.loc[:, "Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    return df


def process_stock_data(
    stocks: List[str], start_date: dt.datetime, end_date: dt.datetime, live: bool
) -> Tuple[dict, List[str]]:
    """Download/load and process price data for all stocks.

    Returns:
        Tuple of (price_dataframes dict, failed_downloads list)
    """
    price_dataframes = {}
    failed_downloads = []
    total_stocks = len(stocks)

    logging.info("--- Downloading Price Data ---")
    for i, stock in enumerate(stocks, 1):
        logging.info(f"Downloading/Loading price data {i}/{total_stocks}: {stock}")

        df = download_or_load_price_data(stock, start_date, end_date, live)
        if df is not None and not df.empty:
            price_dataframes[stock] = df
        else:
            failed_downloads.append(stock)

    return price_dataframes, failed_downloads


def main():
    # Define the time period (e.g., last 90 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=300)

    # Fetch S&P 500 data for RS Rank calculation (as a benchmark)
    index_df = yf.download("^GSPC", start=start_date, end=end_date)

    clean_index_df = clean_price_data(index_df)
    if clean_index_df is None or clean_index_df.empty:
        logging.error("No index data available. Exiting.")
        return

    # Load stock symbols
    tickers = pd.read_csv("tickers.csv", header=None)
    stocks = tickers.iloc[:, 1].tolist()

    price_dataframes, failed_downloads = process_stock_data(
        stocks, start_date, end_date, live=True
    )

    # Create a list to store results
    rs_results = []
    # Add header row
    rs_results.append(["Symbol", "5 Day", "20 Day", "50 Day", "100 Day", "200 Day", "Ratio"])

    for stock, stock_df in price_dataframes.items():
        logging.info(f"Analyzing Relative Strength for {stock}")
        analyze_relative_strength(clean_index_df, stock_df, stock, rs_results)

    # Save results to CSV
    output_file = "relative_strength_results.csv"
    pd.DataFrame(rs_results[1:], columns=rs_results[0]).to_csv(output_file, index=False)
    logging.info(f"Results saved to {output_file}")

    #analyze_rs_deciles("20 Day")
    # Analyze all periods
    analyze_all_periods()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

main()
