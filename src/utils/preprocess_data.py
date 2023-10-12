import pandas as pd
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize

# Download the Punkt tokenizer
nltk.download('punkt')

# Load the raw scheduling data
input_filepath = '../../data/raw/raw_scheduling_data.csv'
raw_data = pd.read_csv(input_filepath)

# Function to convert time to minutes since midnight
def time_to_minutes_since_midnight(time_str):
    time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    return time_obj.hour * 60 + time_obj.minute

# Tokenize station names
raw_data['Tokenized_Station'] = raw_data['Station'].apply(lambda x: word_tokenize(x))

# Convert arrival and departure times to minutes since midnight
raw_data['Arrival_Minutes'] = raw_data['Arrival_Time'].apply(time_to_minutes_since_midnight)
raw_data['Departure_Minutes'] = raw_data['Departure_Time'].apply(time_to_minutes_since_midnight)

# Save the processed data
output_filepath = '../../data/processed/processed_scheduling_data.csv'
raw_data.to_csv(output_filepath, index=False)

print(f"Data preprocessing completed! Processed data saved to {output_filepath}")
