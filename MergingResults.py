import pandas as pd

#Test 2 GPT3.5
test2_GPT35results_1 = pd.read_csv('test2_results_GPT35.csv')
test2_GPT35results_2 = pd.read_csv('test2_results_GPT35_2.csv')

# Merge the two DataFrames
test2_GPT35_final = pd.concat([test2_GPT35results_1, test2_GPT35results_2], ignore_index=True)
print(test2_GPT35_final)

# Save the final DataFrame to a CSV file
test2_GPT35_final.to_csv('test2_GPT35_final.csv', index=False)


#Test 2 GPT4
test2_GPT4_results_1 = pd.read_csv('test2_results_GPT4.csv')
test2_GPT4_results_2 = pd.read_csv('test2_results_GPT4_2.csv')

#Merge
test2_GPT4_final = pd.concat([test2_GPT4_results_1, test2_GPT4_results_2], ignore_index=True)
print(test2_GPT4_final)

# Save the final DataFrame to a CSV file
test2_GPT4_final.to_csv('test2_GPT4_final.csv', index=False)


#Bert Analysis Results 3.5
bertscores_GPT35_1 = pd.read_csv('bertscores_GPT35.csv')
bertscores_GPT35_2 = pd.read_csv('bertscores_GPT35_2.csv')

#Merge & save
bertscores_GPT35_final = pd.concat([bertscores_GPT35_1, bertscores_GPT35_2], ignore_index=True)
print(bertscores_GPT35_final)
bertscores_GPT35_final.to_csv('bertscores_GPT35_final.csv', index=False)


#Bert Analysis Results GPT4
bertscores_GPT4_1 = pd.read_csv('bertscores_GPT4.csv')
bertscores_GPT4_2 = pd.read_csv('bertscores_GPT4_2.csv')

#Merge & save
bertscores_GPT4_final = pd.concat([bertscores_GPT4_1, bertscores_GPT4_2], ignore_index=True)
print(bertscores_GPT4_final)
bertscores_GPT4_final.to_csv('bertscores_GPT4_final.csv', index=False)

