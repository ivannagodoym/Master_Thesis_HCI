import pandas as pd

# Create an empty DataFrame to store the data
test2_results_GPT35_2 = pd.DataFrame(
    columns=['conv_id', 'emotion_label', 'context', 'speaker_utterance', 'original_listener_response'])

#Thesis dataframe
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
thesis_df = pd.read_csv(f"{data_path}/secondsample_test2.csv", sep=",",on_bad_lines='skip')

# Iterate through each conversation in thesis_df
for index, row in thesis_df.iterrows():
    conv_id = row['conv_id']
    emotion_label = row['emotion_label']
    context = row['context']
    speaker_utterance = row['speaker_utterance']
    original_listener_response = row['original_listener_response']

    # Create a new DataFrame with the collected information
    new_row = pd.DataFrame({
        'conv_id': [conv_id],
        'emotion_label': [emotion_label],
        'context': [context],
        'speaker_utterance': [speaker_utterance],
        'original_listener_response': [original_listener_response]
    })

    # Append the new row to the test2_results_GPT35 DataFrame
    test2_results_GPT35_2 = pd.concat([test2_results_GPT35_2, new_row], ignore_index=True)



# Load Test2.1 responses
test21_GPT35_2_responses = pd.read_csv('test21_responses_GPT35_2.csv')
print(test21_GPT35_2_responses)

# Merge Test2.1 responses into test2_data_results based on 'conv_id'
test2_results_GPT35_2 = pd.merge(test2_results_GPT35_2, test21_GPT35_2_responses, on='conv_id', how='left')

# Load Test2.2 responses
test22_GPT35_2_responses = pd.read_csv('test22_responses_GPT35_2.csv')
print(test22_GPT35_2_responses)

# Merge Test2.2 responses into test2_data based on 'conv_id'
test2_results_GPT35_2 = pd.merge(test2_results_GPT35_2, test22_GPT35_2_responses, on='conv_id', how='left')
print(test2_results_GPT35_2)

test2_results_GPT35_2.dropna(axis=1, how='all', inplace=True)
test2_results_GPT35_2.rename(columns={'Test2.1_response': 'Test2.1_response_GPT35', 'Test2.2_response': 'Test2.2_response_GPT35'}, inplace=True)

test2_results_GPT35_2.to_csv('test2_results_GPT35_2.csv', index=False)

# Print the updated DataFrame
print(test2_results_GPT35_2)

