import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv1D, Lambda, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers as layer
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

def gaussian_negative_log_likelihood(y_true, y_pred):
    """
    Gaussian negative log likelihood loss function.
    
    Parameters:
    - y_true: Tensor, true labels.
    - y_pred: Tensor, predictions made by the neural network. The predictions should have shape (n, 2) where the
      first column is the predicted mean (μ) and the second column is the predicted log variance (log(σ²)).
      
    Returns:
    - The negative log likelihood loss value.
    """
    
    # Split the predictions into mean and log variance
    mu, log_variance = tf.split(y_pred, 2, axis=-1)
    
    # Calculate the variance from log variance to ensure positivity
    variance = tf.exp(log_variance)
    
    # Compute the Gaussian negative log likelihood
    loss = 0.5 * (log_variance + tf.square(y_true - mu) / variance) + 0.5 * tf.math.log(2.0 * np.pi)
    
    # Return the mean loss over all samples
    return tf.reduce_mean(loss)

def buildMyNetwork(input_shape):
    inputX = layer.Input(shape=input_shape, dtype='float64')
    num_features = 30
    divP = 2
    layerQ1 = layer.Conv1D(filters = int(num_features/divP), kernel_size = 6 , padding='same')(inputX)
    layerV1 = layer.Conv1D(filters = int(num_features/divP), kernel_size = 6 , padding='same')(inputX)

    attnLayer1 = layer.Normalization()(layer.Attention()([layerQ1, layerV1]))

    #layerQ2 = layer.Conv1D(filters= int(num_features/divP), kernel_size=6, padding='same')(inputX)
    #layerV2 = layer.Conv1D(filters = int(num_features/divP), kernel_size =6 , padding='same')(inputX)

    #attnLayer2 = layer.Normalization()(layer.Attention()([layerQ2, layerV2]))

    attnLayer2 = layer.Dense(num_features/divP, activation='relu')(inputX)

    #nextLayer = layer.GlobalAveragePooling1D()(attnLayer)

    a12 = layer.GlobalAveragePooling1D()(attnLayer1)
    a13 = layer.GlobalAveragePooling1D()(attnLayer2)
    #mergedAttn = tf.math.add(attnLayer1, attnLayer2)
    mergedAttn  = tf.math.multiply(a12, a13)

    nextLayer = layer.Normalization()(mergedAttn)
    final2Hidden = layer.Dense(64, activation='relu')(nextLayer)
    finalHidden = layer.Dense(64,activation='relu')(final2Hidden)
    final23Hidden = layer.Dense(32, activation='relu')(finalHidden)
    final234Hidden = layer.Dense(16, activation='relu')(final23Hidden)
    #kernel = kernels.ExponentiatedQuadratic()

    #y11 = layer.GlobalAveragePooling1D()(final234Hidden)

    output = layer.Dense(2)(final234Hidden)#(y11)



    model = tf.keras.Model(inputs=inputX, outputs=output)
        
    return model
def load_datasets(dataset_dir):
    features = np.load(os.path.join(dataset_dir, "features_2024-03-12_12-20-23.npy"), allow_pickle=True)
    targets = np.load(os.path.join(dataset_dir, "targets_2024-03-12_12-20-23.npy"), allow_pickle=True)
    return features, targets

# Define hyperparameters and paths
window_len = 25
batch_size = 2048
epochs = 2000
checkmark_interval = 10
directory_path = "StockData/2022-01-01-2024-02-08/TrainingData/AAPL_MSFT"  # Update this with your actual directory path
model_save_dir = "StockData/2022-01-01-2024-02-08/Models/AAPL_MSFT" 

os.makedirs(model_save_dir, exist_ok=True)

features, targets = load_datasets(directory_path)
features, targets = shuffle(features, targets, random_state=42)



# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Assuming 'features' and 'targets' are your datasets and have been defined earlier
scale = 100
split_ratio = 0.8
split_index = int(len(features) * split_ratio)

# Scale the features and targets separately
scaled_features = features * scale
scaled_targets = targets * scale

# Split the scaled data into training and validation sets
train_features = scaled_features[:split_index]
train_targets = scaled_targets[:split_index]
val_features = scaled_features[split_index:]
val_targets = scaled_targets[split_index:]


# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Assuming 'features' and 'targets' are your datasets
split_ratio = 0.8
split_index = int(len(features) * split_ratio)

# Concatenate all 2D arrays for fitting the scaler
# This assumes all 2D arrays have the same number of columns (features)
all_train_features = np.concatenate(train_features, axis=0)

# Fit the scaler on all training features at once
scaler.fit(all_train_features)

# Now scale each 2D array in the training and validation features
scaled_train_features = [scaler.transform(features_2d) for features_2d in train_features]
scaled_val_features = [scaler.transform(features_2d) for features_2d in val_features]

# Convert the lists back to arrays if necessary
scaled_train_features = np.array(scaled_train_features)
scaled_val_features = np.array(scaled_val_features)

# Use the scaled features for training and validation
train_features = scaled_train_features
val_features = scaled_val_features
# Now, 'train_features_scaled' and 'val_features_scaled' are your Min-Max scaled features
# 'train_targets_scaled' and 'val_targets_scaled' are your scaled targets, if you choose to scale targets as well

input_shape = train_features[0].shape
model = buildMyNetwork(input_shape)
model.compile(optimizer="adam", loss=gaussian_negative_log_likelihood)

checkpoint_filepath = os.path.join(model_save_dir, 'model_epoch_{epoch:02d}.hdf5')
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, period=checkmark_interval)

model.fit(train_features, train_targets, epochs=epochs, batch_size=batch_size, validation_data=(val_features, val_targets), callbacks=[model_checkpoint_callback])

# Revised part to handle predictions correctly
val_predictions = model.predict(val_features)
predicted_means = val_predictions[:, 0] 
predicted_log_var = val_predictions[:, 1]
predicted_std_devs = np.sqrt(np.exp(predicted_log_var)) 

# Ensure val_targets are on the original scale
unscaled_val_targets = val_targets 

# Calculate the absolute errors
absolute_errors = np.abs(predicted_means - unscaled_val_targets)

# You can then calculate metrics like mean absolute error across all predictions
mean_absolute_error = np.mean(absolute_errors)

# Calculate the 95th and 99th percentiles
percentile_75 = np.percentile(np.abs(absolute_errors), 75)
percentile_90 = np.percentile(np.abs(absolute_errors), 90)
percentile_95 = np.percentile(np.abs(absolute_errors), 95)
percentile_99 = np.percentile(np.abs(absolute_errors), 99)

percentile_95, percentile_99

print(f"Mean Absolute Error: {mean_absolute_error}")
print("Percent: 99: ", percentile_99)
print("Percent: 95: ", percentile_95)
print("Percent: 90: ", percentile_90)
print("Percent: 75: ", percentile_75)

z_95 = 1.96  # Z-score for 95% confidence
z_99 = 2.576  # Z-score for 99% confidence

chunk_size = 100  # Define the chunk size

# Loop through the data in chunks
for i in range(0, len(predicted_means), chunk_size):
    end = i + chunk_size if i + chunk_size <= len(predicted_means) else len(predicted_means)
    
    # Calculate confidence intervals for the current chunk
    lower_bound_95 = predicted_means[i:end] - z_95 * predicted_std_devs[i:end]
    upper_bound_95 = predicted_means[i:end] + z_95 * predicted_std_devs[i:end]
    
    lower_bound_99 = predicted_means[i:end] - z_99 * predicted_std_devs[i:end]
    upper_bound_99 = predicted_means[i:end] + z_99 * predicted_std_devs[i:end]
    
    # Plot for the current chunk
    plt.figure(figsize=(10, 6))
    plt.scatter(range(i, end), unscaled_val_targets[i:end], color='blue', label='Actual')
    plt.scatter(range(i, end), predicted_means[i:end], color='red', label='Predicted', alpha=0.5)
    plt.fill_between(range(i, end), lower_bound_95, upper_bound_95, color='gray', alpha=0.2, label='95% Confidence Interval')
    plt.fill_between(range(i, end), lower_bound_99, upper_bound_99, color='gray', alpha=0.1, label='99% Confidence Interval')
    plt.title(f'Actual vs. Predicted Values with Confidence Intervals (Samples {i+1} to {end})')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # Wait for the plot to be closed before continuing to the next chunk
    plt.close('all')