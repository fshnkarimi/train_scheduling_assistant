import pandas as pd
import random
from datetime import datetime, timedelta

# Set seed for reproducibility
random.seed(0)

# List of sample train IDs
train_ids = [f'T{1000 + i}' for i in range(10)]

# List of sample station names
stations = ['Station_A', 'Station_B', 'Station_C', 'Station_D', 'Station_E']

# Generate synthetic data
data = []

for train_id in train_ids:
    # Generate sequential time data with random delays
    arrival_time = datetime(2023, 10, 10, 6, 0)
    for station in stations:
        delay = random.randint(0, 10)  # Delay in minutes
        arrival_time += timedelta(minutes=random.randint(30, 60) + delay)  # Travel time between stations
        departure_time = arrival_time + timedelta(minutes=random.randint(5, 15))  # Stoppage time at station
        
        data.append([train_id, station, arrival_time, departure_time, delay])
        arrival_time = departure_time  # Set departure time for next station arrival

# Create DataFrame
df = pd.DataFrame(data, columns=['Train_ID', 'Station', 'Arrival_Time', 'Departure_Time', 'Delay'])

# Save synthetic data to CSV
df.to_csv('../../data/raw/raw_scheduling_data.csv', index=False)
