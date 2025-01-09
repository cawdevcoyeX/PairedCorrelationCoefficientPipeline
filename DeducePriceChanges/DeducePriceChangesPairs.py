import os
import pandas as pd
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import cupy as cp


def generate_pairs(lower_bound1, upper_bound1, lower_bound2, upper_bound2, granularity):
    # Calculate step sizes based on granularity
    step_size1 = (upper_bound1 - lower_bound1) / granularity
    step_size2 = (upper_bound2 - lower_bound2) / granularity

    mod = SourceModule("""
    __global__ void generate_pairs(float lower_bound1, float upper_bound1, float step_size1, float lower_bound2, float upper_bound2, float step_size2, float *output, int granularity)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total_steps = (granularity + 1) * (granularity + 1); // Total number of points in the grid

        if (idx < total_steps)
        {
            int i = idx / (granularity + 1); // Determine the current step along the x-axis
            int j = idx % (granularity + 1); // Determine the current step along the y-axis
            
            // Calculate the coordinates for the current point
            float x = lower_bound1 + i * step_size1;
            float y = lower_bound2 + j * step_size2;

            // Cap the values at their respective upper bounds
            x = min(x, upper_bound1);
            y = min(y, upper_bound2);

            // Store the coordinates in the output array
            output[idx * 2] = x;
            output[idx * 2 + 1] = y;
        }
    }
    """)

    # Allocate output array
    output = np.zeros(((granularity + 1) ** 2) * 2, dtype=np.float32)

    # Get the kernel function from the compiled module
    generate_pairs_func = mod.get_function("generate_pairs")

    # Launch the kernel
    generate_pairs_func(
        np.float32(lower_bound1), np.float32(upper_bound1), np.float32(step_size1),
        np.float32(lower_bound2), np.float32(upper_bound2), np.float32(step_size2),
        cuda.Out(output),
        np.int32(granularity),
        block=(1024, 1, 1), grid=((granularity + 1) ** 2 // 1024 + 1, 1)
    )

    # Reshape the output to get pairs of (x, y) coordinates
    return output.reshape(-1, 2)

def read_pairs_data(root_dir):
    pairs_path = os.path.join(root_dir, 'Pairs')
    pairs_data = {}

    for folder in os.listdir(pairs_path):
        folder_path = os.path.join(pairs_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if not '_' in file and file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)
                    pairs_data[folder] = df
                    break  # Assuming only one valid CSV per folder

    return pairs_data

def read_individual_data(root_dir):
    individual_path = os.path.join(root_dir, 'Individual')
    individual_data = {}

    for folder in os.listdir(individual_path):
        folder_path = os.path.join(individual_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('_StdDev.csv'):
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path, index_col=0)
                    individual_data[folder] = df

    return individual_data

# Function to extract bounds for a given stock
def extract_bounds(stock, individual_data, percent):
    if stock in individual_data:
        stock_df = individual_data[stock]
        lower_bound = stock_df.loc[percent, 'Lower Bound']
        upper_bound = stock_df.loc[percent, 'Upper Bound']
        return lower_bound, upper_bound
    else:
        return None, None  # If the stock data is not available

# Function to extract a window of percentage changes for a given stock pair
def extract_window_data(stock1, stock2, idx, window_len, pairs_data):
    key = f"{stock1}_{stock2}"
    
    # Check if the key exists in pairs_data
    if key in pairs_data:
        df = pairs_data[key]  # DataFrame for the stock pair

        # Column names based on stock names
        stock1_col = f"{stock1}_close_pct_change"
        stock2_col = f"{stock2}_close_pct_change"
        
        # Ensure the column names and 'timestamp' exist in the DataFrame
        if stock1_col in df.columns and stock2_col in df.columns and 'timestamp' in df.columns:
            # Calculate start and end indices for the window
            start_idx = max(idx - window_len + 1, 0)  # To avoid negative index
            end_idx = idx  # Dont include the end index
            
            # Extract the data for the window, including the timestamp
            window_data = df.iloc[start_idx:end_idx]
            stock_data = window_data[['timestamp', stock1_col, stock2_col]]
            
            return stock_data
        else:
            print(f"Column names {stock1_col}, {stock2_col}, or 'timestamp' not found in DataFrame")
            return None, None, None
    else:
        print(f"Key {key} not found in pairs_data")
        return None, None, None
    
def calculate_pearson_correlation(x, y):
    """Calculate Pearson correlation coefficient using CuPy."""
    mean_x = cp.mean(x)
    mean_y = cp.mean(y)
    std_x = cp.std(x)
    std_y = cp.std(y)
    correlation = cp.sum((x - mean_x) * (y - mean_y)) / ((len(x) - 1) * std_x * std_y)
    return correlation

def append_pair_and_calculate_correlation(data, pair):
    """Append a pair of values to the data and calculate the correlation."""
    # Convert DataFrame to CuPy array for GPU computation
    data_gpu = cp.asarray(data[[f"{stock1}_close_pct_change", f"{stock2}_close_pct_change"]])
    
    # Append the pair as a new row at the end
    appended_data = cp.vstack([data_gpu, pair])
    
    # Split the data into two series for correlation calculation
    series1 = appended_data[:, 0]  # First stock's data
    series2 = appended_data[:, 1]  # Second stock's data
    
    # Calculate Pearson correlation coefficient
    return calculate_pearson_correlation(series1, series2)

def computePairCorrelations(sequence, pairs):

    cuda_kernel_code = """__global__ void pearson_correlation(float *sequence, float *pairs, float *results, int seq_length, int num_pairs) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_pairs) {
        float x_mean = 0, y_mean = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
        float pair_x = pairs[2*idx]; // Even indices for x
        float pair_y = pairs[2*idx + 1]; // Odd indices for y

        // Calculate means
        for (int i = 0; i < seq_length; i += 2) {
            x_mean += sequence[i];
            y_mean += sequence[i + 1];
        }
        x_mean = (x_mean + pair_x) / (seq_length / 2 + 1);
        y_mean = (y_mean + pair_y) / (seq_length / 2 + 1);

        // Calculate Pearson correlation components
        for (int i = 0; i < seq_length; i += 2) {
            float x = sequence[i] - x_mean;
            float y = sequence[i + 1] - y_mean;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }
        // Include the pair
        sum_xy += (pair_x - x_mean) * (pair_y - y_mean);
        sum_x2 += (pair_x - x_mean) * (pair_x - x_mean);
        sum_y2 += (pair_y - y_mean) * (pair_y - y_mean);

        // Calculate Pearson correlation coefficient
        results[idx] = sum_xy / sqrt(sum_x2 * sum_y2);
    }
}
"""

    # Assume pairs is a list of tuples or a 2D array-like structure
    pairs_flat = np.array(pairs).flatten().astype(np.float32)
    sequence = sequence.astype(np.float32)
    results = np.zeros(len(pairs), dtype=np.float32)

    # Allocate memory on the device
    sequence_gpu = drv.mem_alloc(sequence.nbytes)
    pairs_gpu = drv.mem_alloc(pairs_flat.nbytes)
    results_gpu = drv.mem_alloc(results.nbytes)

    # Copy data to the device
    drv.memcpy_htod(sequence_gpu, sequence)
    drv.memcpy_htod(pairs_gpu, pairs_flat)

    # Compile and get the kernel function
    mod = SourceModule(cuda_kernel_code)
    pearson_corr = mod.get_function("pearson_correlation")

    # Set the block and grid sizes
    block_size = (256, 1, 1)
    grid_size = (int(np.ceil(len(pairs) / 256)), 1)

    # Launch the kernel
    pearson_corr(sequence_gpu, pairs_gpu, results_gpu, np.int32(len(sequence)), np.int32(len(pairs)), block=block_size, grid=grid_size)

    # Copy the results back to host
    drv.memcpy_dtoh(results, results_gpu)

    print("Pearson Correlation Coefficients:", results)

    return results,pairs_flat


# Specify the root directory here
root_dir = 'StockData/2023-01-01-2023-03-01'

# Initialize variables
window_len = 25
granularity = 20
percent = '95%'  # This is a string to match the DataFrame index
error_bound = .1

idx = window_len - 1 

percents = []

# Read data
pairs_data = read_pairs_data(root_dir)
individual_data = read_individual_data(root_dir)

# Iterate over pairs_data keys
for key in pairs_data.keys():
    stock1, stock2 = key.split('_')  # Splitting the key to get individual stock names

    # Extract bounds for Stock1
    stock1_lower, stock1_upper = extract_bounds(stock1, individual_data, percent)
    
    # Extract bounds for Stock2
    stock2_lower, stock2_upper = extract_bounds(stock2, individual_data, percent)

     # Column names based on stock names
    stock1_col = f"{stock1}_close_pct_change"
    stock2_col = f"{stock2}_close_pct_change"
    
    # Here you can do further processing with the extracted bounds
    # For demonstration, just printing them
    print(f"{stock1} Bounds: Lower = {stock1_lower}, Upper = {stock1_upper}")
    print(f"{stock2} Bounds: Lower = {stock2_lower}, Upper = {stock2_upper}")

    correlation = pairs_data[f"{stock1}_{stock2}"].iloc[idx]['pearson_correlation']

    pairs = generate_pairs(stock1_lower, stock1_upper, stock2_lower, stock2_upper, granularity)

    #for idx in range(window_len - 1, len(pairs_data[f"{stock1}_{stock2}"])):

    print("Correlation: ", correlation)
    stock_data = extract_window_data(stock1, stock2, idx, window_len, pairs_data)

    sequence = stock_data[[f"{stock1}_close_pct_change", f"{stock2}_close_pct_change"]].values.flatten()
    
    result, _ = computePairCorrelations(sequence, pairs)

    # Calculate the absolute difference between each element in 'result' and 'correlation'
    differences = np.abs(result - correlation)

    # Count the number of elements where the difference is within the 'error_bound'
    count_within_bounds = np.sum(differences <= error_bound)

    print("Total: " , len(result), ", Within Bound: ", count_within_bounds, ", Percent: ", (count_within_bounds/len(result)))

    percents.append((count_within_bounds/len(result)))


print("Avg percent: ",np.sum(percents)/len(percents))
