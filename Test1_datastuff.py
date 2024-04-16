import pandas as pd

#read the data
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
thesis_data = pd.read_csv(f"{data_path}/thesis_data.csv", sep=",",on_bad_lines='skip')



#
data = pd.read_csv(f"{data_path}/emotion_predictions_gpt35.csv", sep=",",on_bad_lines='skip')

length_data = len(thesis_data)
print(f"Length of data: {length_data}")

conversations = thesis_data.groupby('conv_id')
num_conversations = len(conversations)
print(f"Number of conversations: {num_conversations}")

#drop duplicates
#data = data.drop_duplicates(subset='conv_id')

#update data
#data.to_csv('emotion_predictions_gpt35_noduplicates.csv', index=False)

#create subset of thesis data to test
subset = thesis_data.head(1500)
subset= subset.to_csv('subset.csv', index=False)