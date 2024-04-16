import pandas as pd
#conda remove mkl mkl-service


# Create an empty DataFrame to store the data
test2_data = pd.DataFrame(
    columns=['conv_id', 'emotion_label', 'context', 'speaker_utterance', 'original_listener_response'])

#Thesis dataframe
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
data = pd.read_csv(f"{data_path}/thesis_data.csv", sep=",",on_bad_lines='skip')
thesis_df = pd.DataFrame(data)

# Iterate through each conversation in thesis_df
for conv_id, group in thesis_df.groupby('conv_id'):
    # Extract the emotion label for this conversation
    emotion_label = group['emotion_label'].iloc[0]  # Assuming all utterances in a conversation have the same emotion label :)

    # Extract the context for this conversation
    context = group['context'].iloc[0]  # Assuming the context is the same for all utterances in a conversation

    # Extract the speaker's first utterance
    speaker_utterance = group['utterance'].iloc[0]  # Assuming the speaker's first utterance is always in the first row

    # Extract the first listener's utterance
    first_listener_utterance = group['utterance'].iloc[1] if len(group) >= 2 else 'No utterance'  # If there are fewer than two utterances, set the first utterance as empty string

    # Create a new DataFrame with the collected information
    new_row = pd.DataFrame({
        'conv_id': [conv_id],
        'emotion_label': [emotion_label],
        'context': [context],
        'speaker_utterance': [speaker_utterance],
        'original_listener_response': [first_listener_utterance]
    })

    # Append the new row to the test2_data DataFrame
    test2_data = pd.concat([test2_data, new_row], ignore_index=True)

test2_data.to_csv('test2_data.csv', index=False)