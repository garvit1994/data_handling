# ===============  Outlier detection methods compilation
# ===============  Garvit Bhada - jan 8, 24
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from statsmodels.robust import mad

class OutlierDetector:
    def __init__(self, database_connection_string=None):
        self._db_connection = None
        if database_connection_string:
            self._db_connection = self._connect_to_database(database_connection_string)

    def _connect_to_database(self, connection_string):
        try:
            # Add your code to establish a database connection here
            # Example: db_connection = create_engine(connection_string)
            db_connection = None  # Replace with your actual database connection logic
            return db_connection
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            return None

    def _load_data_from_database(self, table_name, column_name):
        try:
            query = f"SELECT {column_name} FROM {table_name}"
            data = pd.read_sql(query, self._db_connection)
            return data.dropna(subset=[column_name])  # Drop rows with NaN values
        except Exception as e:
            print(f"Error loading data from the database: {e}")
            return pd.DataFrame()

    def _load_data_from_csv(self, csv_path, column_name):
        try:
            data = pd.read_csv(csv_path, usecols=[column_name])
            return data.dropna(subset=[column_name])  # Drop rows with NaN values
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            return pd.DataFrame()

    def _interactive_plot(self, data, outliers, method_name):
        plt.figure(figsize=(12, 6))
        plt.scatter(data.index, data, c=outliers, cmap='viridis', label='Outliers', marker='x')  # Highlight outliers
        plt.title(f'Outliers Detected by {method_name}')
        plt.xlabel('Data Points')
        plt.ylabel('Column Values')
        plt.legend()
        plt.show()

    def _reduce_memory_usage(self, data):
        """
        Reduce memory usage of a Pandas DataFrame by downcasting numeric types.
        """
        numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        for col in data.columns:
            col_type = data[col].dtypes
            if col_type in numerics:
                c_min = data[col].min()
                c_max = data[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        data[col] = data[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        data[col] = data[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        data[col] = data[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        data[col] = data[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        data[col] = data[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        data[col] = data[col].astype(np.float32)
                    else:
                        data[col] = data[col].astype(np.float64)
        return data

    def knn_outlier_detection(self, data, k=5, threshold=1.0):
        try:
            data = data.values.reshape(-1, 1)

            # Scale the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # Apply KNN
            model = NearestNeighbors(n_neighbors=k)
            model.fit(data_scaled)
            distances, _ = model.kneighbors(data_scaled)
            mean_distances = distances.mean(axis=1)

            # Adjust threshold based on the characteristics of your data
            outlier_mask = mean_distances > threshold

            # Highlight outliers in the interactive plot
            self._interactive_plot(data=pd.Series(data.flatten()), outliers=outlier_mask, method_name='KNN')

            return outlier_mask
        except Exception as e:
            print(f"Error in KNN outlier detection: {e}")
            return np.zeros(len(data), dtype=bool)

    def hierarchical_outlier_detection(self, data, threshold=3):
        try:
            data = data.values.reshape(-1, 1)

            # Scale the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # Perform hierarchical clustering
            model = linkage(data_scaled, method='complete', metric='euclidean')

            # Adjust threshold based on the characteristics of your data
            labels = fcluster(model, threshold, criterion='distance')

            # Identify outliers
            outlier_mask = labels != 1

            # Highlight outliers in the interactive plot
            self._interactive_plot(data=pd.Series(data.flatten()), outliers=outlier_mask, method_name='Hierarchical')

            return outlier_mask
        except Exception as e:
            print(f"Error in Hierarchical outlier detection: {e}")
            return np.zeros(len(data), dtype=bool)

    def robust_prior_outlier_detection(self, data, threshold=3):
        try:
            data = data.values
            median = np.nanmedian(data)  # Handle NaN values
            mad_value = mad(data, nan_policy='omit')  # Handle NaN values
            z_scores = np.abs((data - median) / mad_value)
            outlier_mask = z_scores > threshold

            self._interactive_plot(data=pd.Series(data), outliers=outlier_mask, method_name='Robust Prior')

            return outlier_mask
        except Exception as e:
            print(f"Error in Robust Prior outlier detection: {e}")
            return np.zeros(len(data), dtype=bool)

    def isolation_forest_outlier_detection(self, data, contamination=0.05):
        try:
            data = data.values.reshape(-1, 1)
            model = IsolationForest(contamination=contamination)
            outlier_mask = model.fit_predict(data) == -1

            self._interactive_plot(data=pd.Series(data.flatten()), outliers=outlier_mask, method_name='Isolation Forest')

            return outlier_mask
        except Exception as e:
            print(f"Error in Isolation Forest outlier detection: {e}")
            return np.zeros(len(data), dtype=bool)

    def svm_outlier_detection(self, data, nu=0.05):
        try:
            data = data.values.reshape(-1, 1)
            model = OneClassSVM(nu=nu)
            outlier_mask = model.fit_predict(data) == -1

            self._interactive_plot(data=pd.Series(data.flatten()), outliers=outlier_mask, method_name='SVM')

            return outlier_mask
        except Exception as e:
            print(f"Error in SVM outlier detection: {e}")
            return np.zeros(len(data), dtype=bool)

    def autoencoder_outlier_detection(self, data, epochs=50, batch_size=32):
        try:
            data = data.values.reshape(-1, 1)

            # Scale the data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # Define and compile the autoencoder model
            model = keras.Sequential([
                layers.InputLayer(input_shape=(1,)),
                layers.Dense(8, activation='relu'),
                layers.Dense(1, activation='linear')
            ])
            model.compile(optimizer='adam', loss='mse')

            # Train the autoencoder
            history = model.fit(data_scaled, data_scaled, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)

            # Predict and calculate MSE
            reconstructed_data = model.predict(data_scaled)
            mse = np.mean(np.power(data_scaled - reconstructed_data, 2), axis=1)

            # Set the threshold (adjust as needed)
            threshold = np.mean(mse) + 2 * np.std(mse)

            # Identify outliers
            outlier_mask = mse > threshold

            # Highlight outliers in the interactive plot
            self._interactive_plot(data=pd.Series(data.flatten()), outliers=outlier_mask, method_name='Autoencoder')

            return outlier_mask
        except Exception as e:
            print(f"Error in Autoencoder outlier detection: {e}")
            return np.zeros(len(data), dtype=bool)

    def rnn_lstm_outlier_detection(self, data, sequence_length=10, epochs=50, batch_size=32):
        try:
            data = data.values.reshape(-1, 1)
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            def create_sequences(data, sequence_length):
                sequences = []
                for i in range(len(data) - sequence_length + 1):
                    sequence = data[i:i+sequence_length]
                    sequences.append(sequence)
                return np.array(sequences)

            sequences = create_sequences(data_scaled, sequence_length)

            model = keras.Sequential([
                layers.LSTM(8, activation='relu', input_shape=(sequence_length, 1)),
                layers.Dense(1, activation='linear')
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(sequences, sequences, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)

            reconstructed_data = model.predict(sequences)
            mse = mean_squared_error(sequences, reconstructed_data)
            threshold = mse + 2 * np.std(mse)
            outlier_mask = mse > threshold

            self._interactive_plot(data=pd.Series(data.flatten()), outliers=outlier_mask, method_name='RNN LSTM')

            return outlier_mask
        except Exception as e:
            print(f"Error in RNN LSTM outlier detection: {e}")
            return np.zeros(len(data), dtype=bool)

    def detect_outliers(self, data_source, table_name=None, column_name=None, csv_path=None, **kwargs):
        try:
            if data_source == 'database':
                data = self._load_data_from_database(table_name, column_name)
            elif data_source == 'csv':
                data = self._load_data_from_csv(csv_path, column_name)
            else:
                raise ValueError("Invalid data source. Use 'database' or 'csv'.")

            data = self._reduce_memory_usage(data)  # Memory optimization

            knn_outliers = self.knn_outlier_detection(data[column_name], **kwargs.get('knn', {}))
        except Exception as e:
            print(f"Error in KNN outlier detection: {e}")
            knn_outliers = np.zeros(len(data), dtype=bool)

        try:
            hierarchical_outliers = self.hierarchical_outlier_detection(data[column_name], **kwargs.get('hierarchical', {}))
        except Exception as e:
            print(f"Error in Hierarchical outlier detection: {e}")
            hierarchical_outliers = np.zeros(len(data), dtype=bool)

        try:
            robust_prior_outliers = self.robust_prior_outlier_detection(data[column_name], **kwargs.get('robust_prior', {}))
        except Exception as e:
            print(f"Error in Robust Prior outlier detection: {e}")
            robust_prior_outliers = np.zeros(len(data), dtype=bool)

        try:
            isolation_forest_outliers = self.isolation_forest_outlier_detection(data[column_name], **kwargs.get('isolation_forest', {}))
        except Exception as e:
            print(f"Error in Isolation Forest outlier detection: {e}")
            isolation_forest_outliers = np.zeros(len(data), dtype=bool)

        try:
            svm_outliers = self.svm_outlier_detection(data[column_name], **kwargs.get('svm', {}))
        except Exception as e:
            print(f"Error in SVM outlier detection: {e}")
            svm_outliers = np.zeros(len(data), dtype=bool)

        try:
            autoencoder_outliers = self.autoencoder_outlier_detection(data[column_name], **kwargs.get('autoencoder', {}))
        except Exception as e:
            print(f"Error in Autoencoder outlier detection: {e}")
            autoencoder_outliers = np.zeros(len(data), dtype=bool)

        try:
            rnn_lstm_outliers = self.rnn_lstm_outlier_detection(data[column_name], **kwargs.get('rnn_lstm', {}))
        except Exception as e:
            print(f"Error in RNN LSTM outlier detection: {e}")
            rnn_lstm_outliers = np.zeros(len(data), dtype=bool)

        # Display the results
        print("KNN Outliers:", knn_outliers)
        print("Hierarchical Outliers:", hierarchical_outliers)
        print("Robust Prior Outliers:", robust_prior_outliers)
        print("Isolation Forest Outliers:", isolation_forest_outliers)
        print("SVM Outliers:", svm_outliers)
        print("Autoencoder Outliers:", autoencoder_outliers)
        print("RNN LSTM Outliers:", rnn_lstm_outliers)

        return knn_outliers, hierarchical_outliers, robust_prior_outliers, \
               isolation_forest_outliers, svm_outliers, autoencoder_outliers, rnn_lstm_outliers

class OutlierPerformanceMonitor(OutlierDetector):
    def __init__(self, database_connection_string=None):
        super().__init__(database_connection_string)

    def calculate_performance_metrics(self, true_labels, predicted_labels):
        """
        Calculate precision, recall, F1 score, and accuracy.

        Parameters:
        - true_labels: True labels indicating whether each data point is an outlier (ground truth).
        - predicted_labels: Predicted labels indicating whether each data point is an outlier.

        Returns:
        - Dictionary containing precision, recall, F1 score, and accuracy.
        """
        try:
            precision = precision_score(true_labels, predicted_labels)
            recall = recall_score(true_labels, predicted_labels)
            f1 = f1_score(true_labels, predicted_labels)
            accuracy = accuracy_score(true_labels, predicted_labels)

            performance_metrics = {
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'Accuracy': accuracy
            }

            return performance_metrics
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            return None

    def evaluate_performance(self, true_labels, method_name, **kwargs):
        """
        Evaluate the performance of an outlier detection method.

        Parameters:
        - true_labels: True labels indicating whether each data point is an outlier (ground truth).
        - method_name: Name of the outlier detection method.
        - kwargs: Keyword arguments specific to the outlier detection method.

        Returns:
        - Dictionary containing the method name and its corresponding performance metrics.
        """
        try:
            if method_name == 'KNN':
                predicted_labels = self.knn_outlier_detection(**kwargs)
            elif method_name == 'Hierarchical':
                predicted_labels = self.hierarchical_outlier_detection(**kwargs)
            elif method_name == 'Robust Prior':
                predicted_labels = self.robust_prior_outlier_detection(**kwargs)
            elif method_name == 'Isolation Forest':
                predicted_labels = self.isolation_forest_outlier_detection(**kwargs)
            elif method_name == 'SVM':
                predicted_labels = self.svm_outlier_detection(**kwargs)
            elif method_name == 'Autoencoder':
                predicted_labels = self.autoencoder_outlier_detection(**kwargs)
            elif method_name == 'RNN LSTM':
                predicted_labels = self.rnn_lstm_outlier_detection(**kwargs)
            else:
                raise ValueError(f"Invalid method name: {method_name}")

            performance_metrics = self.calculate_performance_metrics(true_labels, predicted_labels)

            return {method_name: performance_metrics}
        except Exception as e:
            print(f"Error evaluating performance for {method_name}: {e}")
            return None

# Example usage:
if __name__ == "__main__":
    # Instantiate the OutlierDetector with the database connection string
    outlier_detector = OutlierDetector(database_connection_string="your_database_connection_string")

    # Load data from the specified table and column
    data_source = 'database'
    table_name = "your_table_name"
    column_name = "your_column_name"

    # Alternatively, load data from a CSV file
    data_source = 'csv'
    csv_path = r"C:\Users\Asus\Downloads\nifty_outlier_detection.csv"
    column_name = "Close"

    outlier_detector.detect_outliers(data_source=data_source, table_name=table_name, column_name=column_name, csv_path=csv_path,
                                     knn={'k': 5, 'threshold': 1.0},
                                     hierarchical={'threshold': 3},
                                     robust_prior={'threshold': 3},
                                     isolation_forest={'contamination': 0.05},
                                     svm={'nu': 0.05},
                                     autoencoder={'epochs': 50, 'batch_size': 32},
                                     rnn_lstm={'sequence_length': 10, 'epochs': 50, 'batch_size': 32})

