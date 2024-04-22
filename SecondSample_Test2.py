import pandas as pd

# Load the thesis data
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
test2_data = pd.read_csv(f"{data_path}/test2_data.csv", sep=",", on_bad_lines='skip')

# Load the first sample of 500 conversations
test2_1sample = pd.read_csv('test2_data_sample500.csv')

def generate_new_sample(first_sample_conv_ids):
    remaining_conv_ids = set(test2_data['conv_id']) - set(first_sample_conv_ids)
    new_sample_conv_ids = pd.Series(list(remaining_conv_ids)).sample(n=500, random_state=42)
    return new_sample_conv_ids


# Generate the first sample of 500 conversations
first_sample_conv_ids = test2_1sample['conv_id'].tolist()

# Generate the second sample of 500 conversations
second_sample_conv_ids = generate_new_sample(first_sample_conv_ids)

# Create the second sample DataFrame
second_sample_test2 = test2_data[test2_data['conv_id'].isin(second_sample_conv_ids)]
print(second_sample_test2)

# Save the second sample to a new CSV file
second_sample_test2.to_csv('secondsample_test2.csv', index=False)
