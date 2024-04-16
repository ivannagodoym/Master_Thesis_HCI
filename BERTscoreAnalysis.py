from bert_score import score
import pandas as pd
import numpy as np

# Load the test2_data DataFrame
test2bert_df = pd.read_csv('test2_data_results.csv')

#adding temporarily: Drop rows with NaN values in either column
test2bert_df.dropna(subset=['original_listener_response', 'Test2.1_response', 'Test2.2_response'], inplace=True)


#original_listener_response and 'Test2.1_response' columns
original_listener_response = test2bert_df['original_listener_response'].tolist()
test21_response = test2bert_df['Test2.1_response'].tolist()
test22_response = test2bert_df['Test2.2_response'].tolist()

# Compute BERTScore for each pair of utterances
bert_scores_21 = []
for utterance1, utterance2 in zip(original_listener_response, test21_response):
    if isinstance(utterance1, str) and isinstance(utterance2, str):  # Check if both utterances are strings
        P, R, F1 = score([utterance1], [utterance2], lang='en', model_type='bert-base-uncased')
        bert_scores_21.append(F1.item())
    else:
        bert_scores_21.append(np.nan)

# Add BERTScore_21 values to a new column
test2bert_df['BERTScore_21'] = bert_scores_21

#Compute BertScore for test22
bert_scores_22 = []
for utterance1, utterance2 in zip(original_listener_response, test22_response):
    if isinstance(utterance1, str) and isinstance(utterance2, str):  # Check if both utterances are strings
        P, R, F1 = score([utterance1], [utterance2], lang='en', model_type='bert-base-uncased')
        bert_scores_22.append(F1.item())
    else:
        bert_scores_22.append(np.nan)

#add BertScore 22
test2bert_df['BERTScore_22'] = bert_scores_22

# Write DataFrame back to CSV file with BERTScore column
test2bert_df.to_csv('Test2_bert_scores.csv', index=False)


