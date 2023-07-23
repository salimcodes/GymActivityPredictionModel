from scipy.signal import butter, filtfilt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


#This class is for Data Processing and Preparation
class Data:
    def __init__(self): 
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler() 

    def drop_unnecessary_columns(self, df, columns_to_drop):
        return df.drop(columns_to_drop, axis=1)
    
    def check_missing_values(self, df):
        return df.isnull().sum()

    def fill_missing_values(self, df):
        return df.fillna(method='ffill') #Missing values will be replaced with the previous non-missing value in the column.

    def scale_data(self, df):
        normalized_data = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns)
        return normalized_data

    def apply_butter_lowpass_filter(self, data, cutoff_frequency, sampling_frequency, order=5):
        # Calculate the normalized cutoff frequency
        nyquist_frequency = 0.5 * sampling_frequency
        normalized_cutoff = cutoff_frequency / nyquist_frequency

        # Create the filter coefficients
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)

        # Apply the filter to each column in the DataFrame
        filtered_data = pd.DataFrame(columns=data.columns)
        for col in data.columns:
            signal = data[col]
            filtered_signal = filtfilt(b, a, signal)
            filtered_data[col] = filtered_signal

        return filtered_data

    def encode_target_labels(self, labels):
        return self.label_encoder.fit_transform(labels)
    
    def preprocess_data(self, df, target_column):
        # Drop unnecessary columns
        df = self.drop_unnecessary_columns(df, ['lux', 'soundLevel'])

        # Handle missing values
        df = self.fill_missing_values(df)

        # Separate features and target
        features = df.drop(target_column, axis=1)
        target = df[target_column]

        # Convert data to float
        features = features.astype(float)

        # Scale the data
        features = self.scale_data(features)

        # Apply low-pass filter
        features = self.apply_butter_lowpass_filter(features, cutoff_frequency=0.1, sampling_frequency=100, order=5)

        # Encode the target labels
        target = self.encode_target_labels(target)

        return features, target

#This class is for Data Analysis
class DataAnalysis:
    def __init__(self, data):
        pass
        
    def calculate_variance(self, data):
        return data.var()
    
    def calculate_standard_deviation(self, data):
        return data.std()

    def calculate_mean(self, data):
        return data.mean()

    def calculate_median(self, data):
        return data.median()

    def calculate_rms(self, data):
        return np.sqrt(np.mean(data**2))

    def calculate_covariance(self, data):
        return data.cov()

    def calculate_zero_crossing(self, data):
        return ((data[:-1] * data[1:]) < 0).sum()

    def calculate_sum_of_squares(self, data):
        return np.sum(data**2)

    def calculate_minimum(self, data):
        return data.min()

    def calculate_maximum(self, data):
        return data.max()

    def calculate_skewness(self, data):
        return skew(data)

    def calculate_kurtosis_(self, data):
        return data.kurtosis()

    def summarize_data_statistics(self, data):
        results = pd.DataFrame()
        for col in data.columns:
            col_data = data[col]
            results[col] = [
                self.calculate_variance(col_data),
                self.calculate_mean(col_data),
                self.calculate_median(col_data),
                self.calculate_standard_deviation(col_data),
                self.calculate_zero_crossing(col_data),
                self.calculate_sum_of_squares(col_data),
                self.calculate_rms(col_data),
                self.calculate_minimum(col_data),
                self.calculate_maximum(col_data),
                self.calculate_skewness(col_data),
                self.calculate_kurtosis_(col_data)
            ]
        results.index = ['variance', 'mean', 'median', 'std', 'nil','zero_crossing', 'sum_of_squares' 'rms', 'minimum', 'maximum', 'skewness', 'kurtosis']
        return results

#This class is for Data Visualization
class DataVisualization:
    def __init__(self, data):
        self.data = data

    def plot_scatter(self, x_column, y_column, xlabel, ylabel, title):
        plt.scatter(self.data[x_column], self.data[y_column])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def plot_histogram(self, data, column, xlabel, ylabel, title):
        plt.hist(data[column])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=90)
        plt.show()

    def plot_correlation(self):
        correlation_matrix = self.data.corr(self.data)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Plot")
        plt.show()

        