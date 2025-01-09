import os
import pandas as pd
import numpy as np
from numba import cuda
import math
from itertools import combinations

def get_time_step(file_name):
    return file_name.split('_')[-1].split('.')[0]

from numba import cuda

@cuda.jit
def pearson_correlation_gpu(x, y, result, slope, intercept, window_length):
    idx = cuda.grid(1)
    if idx >= window_length - 1 and idx < x.size:
        # Initialize sums for mean and square sums
        sum_x, sum_y, sum_xy, square_sum_x, square_sum_y = 0.0, 0.0, 0.0, 0.0, 0.0

        # Calculate sums and square sums for the window
        for i in range(idx - window_length + 1, idx + 1):
            sum_x += x[i]
            sum_y += y[i]
            sum_xy += x[i] * y[i]
            square_sum_x += x[i] ** 2
            square_sum_y += y[i] ** 2

        # Calculate means
        mean_x = sum_x / window_length
        mean_y = sum_y / window_length

        # Calculate variances and covariance
        variance_x = square_sum_x - window_length * mean_x ** 2
        variance_y = square_sum_y - window_length * mean_y ** 2
        covariance = sum_xy - window_length * mean_x * mean_y

        # Calculate slope (m) and y-intercept (b) for linear regression
        if variance_x > 0:
            slope[idx] = covariance / variance_x
            intercept[idx] = mean_y - slope[idx] * mean_x

            # Calculate Pearson correlation coefficient if both variances are non-zero
            if variance_y > 0:
                result[idx] = covariance / (math.sqrt(variance_x) * math.sqrt(variance_y))
            else:
                result[idx] = 0.0  # Set correlation to 0 if variance_y is 0
        else:
            # Handle the case where variance_x is 0
            slope[idx] = 0.0
            intercept[idx] = mean_y  # Use mean_y as the intercept
            result[idx] = 0.0  # Set correlation to 0 if variance_x is 0
    elif idx < window_length - 1:
        # For indices smaller than window_length - 1, set all to 0
        result[idx] = slope[idx] = intercept[idx] = 0.0


def calculate_pearson_correlation(directory, output_root, window_length=25):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    symbols = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    for symbol1, symbol2 in combinations(symbols, 2):
        symbol1_dir = os.path.join(directory, symbol1)
        symbol2_dir = os.path.join(directory, symbol2)
        symbol1_files = [f for f in os.listdir(symbol1_dir) if f.endswith('.csv')]
        symbol2_files = [f for f in os.listdir(symbol2_dir) if f.endswith('.csv')]

        for file1 in symbol1_files:
            if 'StdDev' in file1:
                continue
            time_step1 = get_time_step(file1)
            for file2 in symbol2_files:
                if 'StdDev' in file2:
                    continue
                time_step2 = get_time_step(file2)
                if time_step1 == time_step2:
                    df1 = pd.read_csv(os.path.join(symbol1_dir, file1))
                    df2 = pd.read_csv(os.path.join(symbol2_dir, file2))

                    stock1_prefix = symbol1 + '_'
                    stock2_prefix = symbol2 + '_'

                    df1.rename(columns=lambda col: stock1_prefix + col if col != 'timestamp' else col, inplace=True)
                    df2.rename(columns=lambda col: stock2_prefix + col if col != 'timestamp' else col, inplace=True)

                    merged_df = pd.merge(df1, df2, on='timestamp')
                    merged_df.dropna(inplace=True)

                    x = np.array(merged_df[symbol1 + '_close_pct_change'], dtype=np.float64)
                    y = np.array(merged_df[symbol2 + '_close_pct_change'], dtype=np.float64)
                    n = len(x)

                    x_device = cuda.to_device(x)
                    y_device = cuda.to_device(y)
                    result_device = cuda.device_array(n, dtype=np.float64)
                    m_device = cuda.device_array(n, dtype=np.float64)
                    b_device = cuda.device_array(n, dtype=np.float64)

                    threads_per_block = 512
                    blocks = math.ceil(n / threads_per_block)

                    pearson_correlation_gpu[blocks, threads_per_block](x_device, y_device, result_device, m_device, b_device, window_length)

                    correlation_coeffs = result_device.copy_to_host()
                    slope = m_device.copy_to_host()
                    intercept = b_device.copy_to_host()

                    merged_df['slope'] = slope
                    merged_df['intercept'] = intercept
                    merged_df['pearson_correlation'] = correlation_coeffs

                    output_dir = os.path.join(output_root, f"{symbol1}_{symbol2}")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    output_path = os.path.join(output_dir, time_step1 + '.csv')
                    merged_df.to_csv(output_path, index=False)

                    print(f"Processed and saved: {output_path}")

directory = 'StockData/2022-01-01-2024-02-08/Individual'
output_root = 'StockData/2022-01-01-2024-02-08/Pairs'
window_length = 25
calculate_pearson_correlation(directory, output_root, window_length)
