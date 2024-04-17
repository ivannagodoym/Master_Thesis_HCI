import pandas as pd

#read the data
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
test1_data = pd.read_csv(f"{data_path}/EmotionPrediction_GPT35.csv", sep=",",on_bad_lines='skip')

#drop conv_id duplicates
test1_data = test1_data.drop_duplicates(subset='conv_id')
#print(test1_data)

#update data
test1_data.to_csv('EmotionPrediction_GPT35_nodup.csv', index=False)

#Data Analytics

# Count occurrences where the original labeled emotion is present and is top 1
top_1_count = test1_data['position_of_original_label'].eq(1).sum()

# Count occurrences where the original labeled emotion is present in top 3
top_3_count = test1_data['position_of_original_label'].notnull().sum()

# Calculate percentages
total_rows = len(test1_data)
percentage_top_3 = (top_3_count / total_rows) * 100
percentage_top_1 = (top_1_count / total_rows) * 100
print("Total rows:", total_rows)
print("Top 1 count:", top_1_count)
print("Top 3 count:", top_3_count)
print(percentage_top_1)
print(percentage_top_3)

print("Percentage of times original labeled emotion appeared in top 3 responses:", "{:.2f}%".format(percentage_top_3))
print("Percentage of times original labeled emotion appeared as top 1 response:", "{:.2f}%".format(percentage_top_1))

