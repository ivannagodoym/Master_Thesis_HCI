import pandas as pd

#read the data
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
thesis_data = pd.read_csv(f"{data_path}/thesis_data.csv", sep=",",on_bad_lines='skip')

length_data = len(thesis_data)

conversations = thesis_data.groupby('conv_id')
num_conversations = len(conversations)


# Divide conversations into three parts
conversations_list = list(conversations)
part1 = thesis_data[:17473]
#part2 = conversations_list[4000:7000]
#part3 = conversations_list[7000:]

# Save each part into separate files
#part1_df = pd.concat([conv[1] for conv in part1])
#part2_df = pd.concat([conv[1] for conv in part2])
#part3_df = pd.concat([conv[1] for conv in part3])

part1.to_csv("thesis_data1.csv", index=False)
#part2_df.to_csv("thesis_data2.csv", index=False)
#part3_df.to_csv("thesis_data3.csv", index=False)

#print("Data saved into thesis_data1.csv, thesis_data2.csv, and thesis_data3.csv.")

#Get a sample of 500 conversations
sampled_conv_ids = conversations['conv_id'].unique().sample(n=500, random_state=42)

# Select all rows corresponding to sampled conversation IDs
test2_data = thesis_data[thesis_data['conv_id'].isin(sampled_conv_ids.explode())]
print(test2_data)
print(test2_data.shape)

test2_data.to_csv("test2_data.csv", index=False)