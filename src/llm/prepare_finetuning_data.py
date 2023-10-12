import pandas as pd

# Load the processed data
input_filepath = '../../data/processed/processed_scheduling_data.csv'
processed_data = pd.read_csv(input_filepath)

# Create sentences from the processed data
# Example: 'Train T1000 arrives at Station_A at 360 minutes with a delay of 5 minutes.'
processed_data['sentence'] = 'Train ' + processed_data['Train_ID'].astype(str) + \
                             ' arrives at ' + processed_data['Tokenized_Station'].astype(str) + \
                             ' at ' + processed_data['Arrival_Minutes'].astype(str) + \
                             ' minutes with a delay of ' + processed_data['Delay'].astype(str) + ' minutes.'

# Save sentences to a text file for fine-tuning
with open('../../data/processed/finetuning_data.txt', 'w', encoding='utf-8') as file:
    for sentence in processed_data['sentence']:
        file.write(sentence + '\n')

print("Data preparation for fine-tuning completed!")
