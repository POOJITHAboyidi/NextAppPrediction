import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, BatchNormalization, Dense
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
import os
from sklearn.metrics import accuracy_score, mean_squared_error

logging.basicConfig(level=logging.DEBUG)

def clean_data(data):
    logging.info(f"Original data length: {len(data)}")
    
    # Define exact patterns to remove
    unwanted_exact_phrases = [
        "Activity history, January 2, 2019 - December 27, 2018",
        "Created by App Usage (PRO) on Wednesday, January 2, 2019, 20:21"
    ]
    
    # Remove rows containing these exact phrases
    mask = ~data.iloc[:, 0].isin(unwanted_exact_phrases)
    cleaned_data = data[mask]
    
    logging.info(f"Cleaned data length: {len(cleaned_data)}")
    return cleaned_data

def encode_data(data):
    label_encoder_app = LabelEncoder()
    encoded_data = label_encoder_app.fit_transform(data.iloc[:, 0])
    encoded_data = pd.DataFrame(encoded_data)
    return encoded_data, label_encoder_app

def build_model():
    model = Sequential()
    model.add(LSTM(200, return_sequences=True, input_shape=(10, 1)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(LSTM(200, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(LSTM(200, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(36, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Path to the dataset
file_path = r'C:/Users/welcome/Desktop/CDAC_PROJECT1/Dataset/dataset.csv'

# Check if file exists and is accessible
if os.path.isfile(file_path):
    logging.info(f"File found: {file_path}")
else:
    logging.error(f"File not found: {file_path}")

if os.access(file_path, os.R_OK):
    logging.info(f"File is readable: {file_path}")
else:
    logging.error(f"File is not readable: {file_path}")

try:
    # Load the data
    data = pd.read_csv(file_path)
    logging.info(f"Data loaded successfully with {len(data)} rows.")

    # Clean data
    data = clean_data(data)
    logging.info(f"Data after cleaning has {len(data)} rows.")

    # Encode data
    encoded_data, label_encoder_app = encode_data(data)
    logging.info(f"Data encoding completed. Encoded data has {len(encoded_data)} rows.")

    # Prepare sequential data for LSTM
    total_dataset = encoded_data.iloc[:, 0:1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(total_dataset)

    X = []
    y = []
    for i in range(10, len(scaled_data)):
        X.append(scaled_data[i-10:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    logging.info(f"Data prepared with {X.shape[0]} sequences.")

    # One-hot encode the target variable
    y_one_hot = to_categorical(y, num_classes=36)

    # K-Fold Cross Validation
    kfold = KFold(n_splits=5, shuffle=True)
    fold_no = 1
    accuracies = []
    rmses = []
    all_predictions = []

    for train, test in kfold.split(X, y_one_hot):
        logging.info(f'Training fold {fold_no}')
        model = build_model()

        # Train the model
        model.fit(X[train], y_one_hot[train], epochs=5, batch_size=32, verbose=1)

        # Evaluate the model
        predicted_app = model.predict(X[test])
        idx = (-predicted_app).argsort(axis=1)[:, :4]  # Get indices of top 4 predictions

        # Actual apps used
        actual_app_used = np.argmax(y_one_hot[test], axis=1)

        # Calculate top-1 accuracy
        top1_predictions = idx[:, 0]
        top1_accuracy = accuracy_score(actual_app_used, top1_predictions)
        accuracies.append(top1_accuracy)
        logging.info(f"Fold {fold_no} - Top-1 Accuracy: {top1_accuracy * 100:.2f}%")

        # Calculate RMSE
        actual_indices = actual_app_used
        predicted_probs = np.array([predicted_app[i, actual_indices[i]] for i in range(len(actual_indices))])
        rmse = np.sqrt(mean_squared_error(np.ones(len(actual_indices)), predicted_probs))
        rmses.append(rmse)
        logging.info(f"Fold {fold_no} - RMSE: {rmse * 100}")

        # Store predictions
        fold_predictions = pd.DataFrame({
            'Prediction1': label_encoder_app.inverse_transform(idx[:, 0]),
            'Prediction2': label_encoder_app.inverse_transform(idx[:, 1]),
            'Prediction3': label_encoder_app.inverse_transform(idx[:, 2]),
            'Prediction4': label_encoder_app.inverse_transform(idx[:, 3]),
            'Actual App Used': label_encoder_app.inverse_transform(actual_app_used)
        })
        all_predictions.append(fold_predictions)

        fold_no += 1

    logging.info(f'Average Top-1 Accuracy: {np.mean(accuracies) * 100:.2f}%')
    logging.info(f'Average RMSE: {np.mean(rmses) * 100}')

    # Aggregate predictions across all folds
    combined_predictions = pd.concat(all_predictions, ignore_index=True)

    # Calculate average predictions
    average_predictions = combined_predictions.groupby(combined_predictions.index).agg(lambda x: x.mode().iloc[0])
    print('AVERAGE PREDICTIONS')
    print(average_predictions)

except Exception as e:
    logging.error(f"Error processing data: {e}")

