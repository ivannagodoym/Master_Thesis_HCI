from bert_score import score
import pandas as pd
import numpy as np

# Load the test2_data DataFrame
bertscores_GPT35 = pd.read_csv('test2_results_GPT35_2.csv')

# original_listener_response and 'Test2.1_response' columns
original_listener_response = bertscores_GPT35['original_listener_response'].tolist()
test21_response_GPT35 = bertscores_GPT35['Test2.1_response_GPT35'].tolist()
test22_response_GPT35 = bertscores_GPT35['Test2.2_response_GPT35'].tolist()


# Compute BERTScore for each pair of utterances
bertscores_21 = []
for utterance1, utterance2 in zip(original_listener_response, test21_response_GPT35):
    if isinstance(utterance1, str) and isinstance(utterance2, str):  # Check if both utterances are strings
        P, R, F1 = score([utterance1], [utterance2], lang='en', model_type='bert-base-uncased')
        bertscores_21.append(F1.item())
    else:
        bertscores_21.append(np.nan)

# Add BERTScore_21 values to a new column
bertscores_GPT35['BERTScore_Test21_GPT35'] = bertscores_21


# Compute BertScore for test22
bertscores_22 = []
for utterance1, utterance2 in zip(original_listener_response, test22_response_GPT35):
    if isinstance(utterance1, str) and isinstance(utterance2, str):  # Check if both utterances are strings
        P, R, F1 = score([utterance1], [utterance2], lang='en', model_type='bert-base-uncased')
        bertscores_22.append(F1.item())
    else:
        bertscores_22.append(np.nan)

#add BertScore 22
bertscores_GPT35['BERTScore_Test22_GPT35'] = bertscores_22

# Write DataFrame back to CSV file with BERTScore column
bertscores_GPT35.to_csv('bertscores_GPT35_2.csv', index=False)


