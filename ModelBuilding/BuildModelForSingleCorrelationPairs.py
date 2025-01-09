import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras import layers as layer
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

def min95Percent(y_true, y_pred):
    # Convert inputs to float64
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    per = tf.math.abs(y_true - y_pred)
    mean = tf.math.reduce_mean(per)
    samples = tf.cast(tf.size(y_true), tf.float64)  # Also convert sample size to float64
    std_dev = tf.math.sqrt(tf.math.reduce_sum((per - mean) ** 2))
    std_err = std_dev / tf.math.sqrt(samples)

    z_score = 1.645  # 1.96 for 95%, 1.645 for 90%
    error_margin = z_score * std_err

    lower = mean - error_margin
    upper = mean + error_margin

    return error_margin  # Return the absolute error margin

def min_percent_threshold(y_true, y_pred, percent):
    # Calculate the absolute differences
    abs_diff = np.abs(np.array(y_true) - np.array(y_pred))
    
    # Sort the absolute differences
    sorted_diff = np.sort(abs_diff)
    
    # Calculate the index for the desired percentile
    index = int(np.ceil(percent / 100.0 * len(sorted_diff))) - 1
    
    # Ensure the index is within the bounds of the array
    index = max(min(index, len(sorted_diff) - 1), 0)
    
    # Find the value at the calculated index
    threshold = sorted_diff[index]
    
    return threshold

def load_data(full_path):

    # Find the CSV file without an underscore in its name
    for file in os.listdir(full_path):
        if file.endswith('.csv') and '_' not in file:
            data_path = os.path.join(full_path, file)
            break

    # Load the data
    data = pd.read_csv(data_path)

    #data.drop(columns=['timestamp'], inplace=True)
    #data.dropna(axis=0, inplace=True)

    return data

def convert_timestamps(data):
    # Convert string timestamps to datetime objects, assuming the format includes timezone information
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Normalize timestamps to UTC
    data['timestamp'] = data['timestamp'].dt.tz_convert('UTC')

    # Convert timestamps to a numeric scale between 0 and 1
    min_timestamp = data['timestamp'].min()
    max_timestamp = data['timestamp'].max()
    timestamp_range = (max_timestamp - min_timestamp).total_seconds()

    # Apply Min-Max scaling to scale timestamps between 0 and 1
    data['timestamp_numeric'] = data['timestamp'].apply(lambda x: ((x - min_timestamp).total_seconds()) / timestamp_range)

    return data


def scale_time_of_day(dataframe):
    # Ensure 'timestamp' column is in datetime format
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
    
    # Define start and end times (9 AM and 4 PM)
    start_time = pd.to_datetime('09:00:00').time()
    end_time = pd.to_datetime('16:00:00').time()

    # Calculate the total seconds between 9 AM and 4 PM
    total_seconds = (pd.Timestamp.combine(pd.to_datetime('today'), end_time) - 
                     pd.Timestamp.combine(pd.to_datetime('today'), start_time)).total_seconds()

    # Calculate the seconds since 9 AM for each timestamp and scale between 0 and 1
    dataframe['time_of_day'] = dataframe['timestamp'].apply(
        lambda x: ((x.time().hour * 3600 + x.time().minute * 60 + x.time().second) - 
                   (start_time.hour * 3600 + start_time.minute * 60 + start_time.second)) / total_seconds
    )

    # Ensure that all values are within the 0 to 1 range
    dataframe['time_of_day'] = dataframe['time_of_day'].clip(lower=0, upper=1)

    return dataframe


def create_dataset(data, window_len):
    features = []
    targets = []

    data = scale_time_of_day(data)

    # Convert timestamps to numeric values
    data = convert_timestamps(data)

    # Ensure required columns are present
    required_columns = ['slope', 'intercept', 'pearson_correlation']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"The data does not contain the '{col}' column.")

    for i in range(len(data) - window_len):
        # Select 'slope', 'intercept', 'pearson_correlation', and converted 'timestamp' columns for features
        slope = data.iloc[i:i+window_len]['slope'].values.reshape(-1, 1)
        intercept = data.iloc[i:i+window_len]['intercept'].values.reshape(-1, 1)
        pearson_corr = data.iloc[i:i+window_len]['pearson_correlation'].values.reshape(-1, 1)
        timestamp_numeric = data.iloc[i:i+window_len]['timestamp_numeric'].values.reshape(-1, 1)
        time_of_day = data.iloc[i:i+window_len]["time_of_day"].values.reshape(-1,1)

        # Stack arrays horizontally to create a single feature array for each sample
        feature_array = np.hstack((slope, intercept, pearson_corr, timestamp_numeric, time_of_day))
        features.append(feature_array)
        targets.append(data.iloc[i+window_len]['pearson_correlation'])  # Use 'pearson_correlation' column as the target

    features = np.array(features)
    targets = np.array(targets)

    return features, targets


def create_randomized_dataset(data, window_len):
    features = []
    targets = []

    data = scale_time_of_day(data)

    # Convert timestamps to numeric values
    data = convert_timestamps(data)

    num_samples = len(data) - window_len

    # Assuming 'pearson_correlation' is the target and should not be duplicated in the features
    feature_columns = [col for col in data.columns if col != 'timestamp']

    # Assuming 'pearson_correlation' is the target and should not be duplicated in the features
    scale_columns = [col for col in data.columns if col not in ['timestamp', 'pearson_correlation', 'intercept', 'slope']]

    # Initialize MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the data for scale_columns and update these columns in data
    data[scale_columns] = scaler.fit_transform(data[scale_columns])

    for i in range(0, num_samples):
        # Select the transformed features for the current window
        window_data = data.iloc[i:i+window_len][feature_columns]
        
        # Convert window data to numpy array and flatten to 1D per sample
        feature_array = window_data.values.reshape(window_data.shape[0], -1)
        features.append(feature_array)

        # Assuming the target is the value of 'pearson_correlation' at the end of the window
        targets.append(data.iloc[i+window_len]['pearson_correlation'])


    features = np.array(features)
    targets = np.array(targets)

    # Shuffle features and targets in unison
    features, targets = shuffle(features, targets, random_state=42)

    return features, targets




class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.multi_head_attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(input_shape, num_heads, ff_dim, num_layers, mlp_units, dropout_rate=0.1):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_layers):
        x = TransformerEncoderLayer(embed_dim=input_shape[-1], num_heads=num_heads, ff_dim=ff_dim, rate=dropout_rate)(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    for units in mlp_units:
        x = Dense(units, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)  # Assuming a regression task for the Pearson correlation prediction

    model = Model(inputs=inputs, outputs=outputs)
    return model

def buildMyNetwork(input_shape, num_heads, ff_dim, num_layers, mlp_units, dropout_rate=0.1):
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

    output = layer.Dense(1)(final234Hidden)#(y11)



    model = tf.keras.Model(inputs=inputX, outputs=output)
        
    return model


# Define your hyperparameters
window_len = 25  # Adjust this as needed
num_heads = 5
ff_dim = 30
num_layers = 6
mlp_units = [  128, 64]
dropout_rate = 0.1
batch_size = 256
epochs = 5 # Adjust this as needed

# Specify the directory path and stock names
directory_path = "StockData/2023-01-01-2024-02-08/Pairs/AAPL_MSFT"  # Update this with your actual directory path

# Load the data
data = load_data(directory_path)

# Create the dataset
features, targets = create_randomized_dataset(data, window_len)

# Split the data into training and validation sets
split_ratio = 0.8  # You can adjust this ratio
split_index = int(len(features) * split_ratio)

train_features = features[:split_index]
train_targets = targets[:split_index]
val_features = features[split_index:]
val_targets = targets[split_index:]

# Build the transformer model
input_shape = (window_len, 35 )  # Input shape based on data columns
model = buildMyNetwork(input_shape, num_heads, ff_dim, num_layers, mlp_units, dropout_rate)
#model = build_transformer_model(input_shape, num_heads, ff_dim, num_layers, mlp_units, dropout_rate)

# Compile the model
model.compile(optimizer="adam", loss = "mean_squared_error")#loss=min95Percent)

model.summary()
# Train the model
history = model.fit(train_features, train_targets, epochs=epochs, batch_size=batch_size, validation_data=(val_features, val_targets))

# Make predictions on the validation data
val_predictions = model.predict(val_features)
val_predictions = val_predictions.flatten()

val_targets = val_targets 
val_predictions = val_predictions 


absolute_differences = np.abs((val_targets) - (val_predictions))

# Calculate the average of the absolute differences
average_absolute_difference = np.mean(absolute_differences)

print("Absolute Difference: ", average_absolute_difference)
bounds = min_percent_threshold(val_targets, val_predictions, 95)
print("Error 95%: ", bounds)

bounds = min_percent_threshold(val_targets, val_predictions, 90)
print("Error 90%: ", bounds)
# Optionally, you can calculate and print other metrics like RMSE, MAE, etc.
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(val_targets, val_predictions)
# ...

# You can also visualize the predictions vs. actual values for further analysis
# For example, using matplotlib:

# Create a figure and axes
fig, ax = plt.subplots()

yeqx1 = np.linspace(-1, 1, 100) 
yeqx2 = yeqx1
# Set the x-axis limits
ax.set_xlim(-1, 1)
ax.plot(yeqx1, yeqx2, color='orange', label='y = x')

# Set the y-axis limits
ax.set_ylim(-1, 1)
plt.scatter(val_targets, val_predictions)
plt.xlabel("Actual Pearson Correlation")
plt.ylabel("Predicted Pearson Correlation")
plt.title("Actual vs. Predicted Pearson Correlation")
plt.show()