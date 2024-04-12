from bert_score import score
import pandas as pd
import numpy as np

# Load the test2_data DataFrame
test2bert_df = pd.read_csv('test2_data.csv')

#adding temporarily: Drop rows with NaN values in either column
test2bert_df.dropna(subset=['original_listener_response', 'Test2.1_response'], inplace=True)

#original_listener_response and 'Test2.1_response' columns
original_listener_response = test2bert_df['original_listener_response'].tolist()
test21_response = test2bert_df['Test2.1_response'].tolist()

# Compute BERTScore for each pair of utterances
bert_scores = []
for utterance1, utterance2 in zip(original_listener_response, test21_response):
    if isinstance(utterance1, str) and isinstance(utterance2, str):  # Check if both utterances are strings
        P, R, F1 = score([utterance1], [utterance2], lang='en', model_type='bert-base-uncased')
        bert_scores.append(F1.item())
    else:
        bert_scores.append(np.nan)

# Add BERTScore values to a new column
test2bert_df['BERTScore'] = bert_scores

# Write DataFrame back to CSV file with BERTScore column
test2bert_df.to_csv('Test2_bert_scores.csv', index=False)


