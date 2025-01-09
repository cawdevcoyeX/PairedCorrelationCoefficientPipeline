import os
import glob
import pandas as pd
import numpy as np

def process_order_book(df_csv, symbol, order_book_file, changes_dir):
    # Read order book data
    df_order_book = pd.read_csv(order_book_file)

    # Adjust prices
    df_order_book['bid_px_00'] /= 1e9
    df_order_book['ask_px_00'] /= 1e9

    # Convert timestamp and adjust timezone
    df_order_book['ts_event_datetime'] = pd.to_datetime(df_order_book['ts_event'], unit='ns', utc=True)
    df_order_book['ts_event_datetime'] = df_order_book['ts_event_datetime'].dt.tz_convert('America/New_York')

    # Filter for trading hours
    df_order_book.set_index('ts_event_datetime', inplace=True)
    df_order_book = df_order_book.between_time('09:30', '16:00').reset_index()

    # Generate minute-level summary
    df_order_book['timestamp'] = df_order_book['ts_event_datetime'].dt.floor('T')
    minute_summary = df_order_book.groupby(['symbol', 'timestamp']).agg(
        closing_bid_px=('bid_px_00', 'last'),
        closing_ask_px=('ask_px_00', 'last'),
        total_trades=('size', lambda x: df_order_book.loc[x.index, 'size'][df_order_book.loc[x.index, 'action'] == 'T'].sum())
    ).reset_index()

    # Filter minute_summary for the given symbol
    minute_summary_filtered = minute_summary[minute_summary['symbol'] == symbol]

    # Ensure 'timestamp' in df_csv is the correct dtype and tz-aware
    df_csv['timestamp'] = pd.to_datetime(df_csv['timestamp']).dt.tz_localize(None).dt.tz_localize('America/New_York')

    # Merge and update
    # Perform a left merge to join the dataframes on the 'timestamp' column.
    merged_df = pd.merge(df_csv, minute_summary_filtered, on='timestamp', how='left', suffixes=('', '_right'))

    # For each column that exists in both dataframes, update NaN values in the left dataframe with values from the right.
    for column in set(df_csv.columns).intersection(set(minute_summary_filtered.columns)) - {'timestamp'}:
        merged_df[column] = merged_df[column].combine_first(merged_df[column + '_right'])

    # Drop the duplicate columns from the right dataframe.
    merged_df.drop(columns=[col + '_right' for col in minute_summary_filtered.columns if col != 'timestamp'], inplace=True, errors='ignore')

    # Update the original df_csv dataframe
    df_csv = merged_df

    df_csv.drop('symbol', axis=1, inplace=True)
    #df_csv.update(merged_data.set_index('timestamp'), overwrite=True)
    

    # Save the changes to a CSV file
    #changes_file_path = os.path.join(changes_dir, os.path.basename(order_book_file).replace('.csv', '_changes.csv'))
    #df_csv.to_csv(changes_file_path, index=False)
    return df_csv

def process_stock_files(directory_path, order_book_dir, stock1, stock2):
    stock1_dir = os.path.join(directory_path, stock1)
    stock2_dir = os.path.join(directory_path, stock2)

    # Directory to save changes files
    changes_dir = os.path.join(directory_path, 'changes')
    os.makedirs(changes_dir, exist_ok=True)

    for stock_dir, symbol in [[stock1_dir, stock1], [stock2_dir, stock2]]:
        csv_files = glob.glob(os.path.join(stock_dir, '[!StdDev]*.csv'))

        if symbol == 'AAPL':
            continue
        
        for csv_file in csv_files:
            if not 'Min' in csv_file or 'StdDev' in csv_file:
                continue
            print(f"Processing CSV file: {csv_file}")
            df_csv = pd.read_csv(csv_file)

            # Initialize 'closing_bid_px' and 'closing_ask_px' in df_csv with NaN


            order_book_files = glob.glob(os.path.join(order_book_dir, 'xnas-itch-*.csv'))

            for order_book_file in order_book_files:
                print(f"Using order book file: {order_book_file}")
                df_csv = process_order_book(df_csv, symbol, order_book_file, changes_dir)

            # Save the updated df_csv
            output_csv_path = os.path.join(stock_dir, f"{symbol}_updated.csv")
            df_csv.to_csv(output_csv_path, index=False)
            print(f"Updated data saved to {output_csv_path}")   

# Example usage
directory_path = 'StockData/2022-01-01-2024-02-08/Individual'
order_book_dir = 'XNAS-20240310-EF7MMQ67CR'
stock1 = 'AAPL'
stock2 = 'MSFT'
process_stock_files(directory_path, order_book_dir, stock1, stock2)
