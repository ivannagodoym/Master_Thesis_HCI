import pandas as pd

test2_results_GPT4 = pd.read_csv('test2_results_GPT4.csv')
#Get random 100 samples
#comparing_responses_sample = test2_results_GPT4.sample(n=100, random_state=42)
#print(comparing_responses_sample)

#drop columns that are not needed
#comparing_responses_sample.drop(columns=['emotion_label', 'context', 'original_listener_response'], inplace=True)


#comparing_responses_sample.to_csv('comparing_responses_sample.csv', index=False)

# Load the comparing_responses_sample.csv file
comparing_responses_sample = pd.read_csv('comparing_responses_sample.csv')

# Store the conversation IDs of the initial 100 rows
initial_conv_ids = comparing_responses_sample['conv_id']

# Get additional 30 rows excluding the conversation IDs of the initial sample
additional_rows = test2_results_GPT4[~test2_results_GPT4['conv_id'].isin(initial_conv_ids)].sample(n=20, random_state=42)

# Append the additional rows to the initial sample
extended_comparing_responses_sample = pd.concat([comparing_responses_sample, additional_rows])
extended_comparing_responses_sample.drop(columns=['emotion_label', 'context', 'original_listener_response'], inplace=True)


#drop rows: hit:9059_conv:18118,hit:6529_conv:13059, hit:5406_conv:10812,hit:7670_conv:15341,hit:651_conv:1303,
# hit:7783_conv:15567,hit:5168_conv:10337,hit:4855_conv:9711,hit:126_conv:253, hit:11488_conv:22976, hit:5758_conv:11517, hit:1853_conv:3707,
#hit:6055_conv:12111, hit:5830_conv:11660, hit:6311_conv:12622, hit:6564_conv:13129, hit:525_conv:1051, hit:903_conv:1806,
#hit:2705_conv:5410, hit:5406_conv:10812
conv_ids_to_drop = [
    'hit:9059_conv:18118', 'hit:6529_conv:13059', 'hit:5406_conv:10812', 'hit:7670_conv:15341',
    'hit:651_conv:1303', 'hit:7783_conv:15567', 'hit:5168_conv:10337', 'hit:4855_conv:9711',
    'hit:126_conv:253', 'hit:11488_conv:22976', 'hit:5758_conv:11517', 'hit:1853_conv:3707',
    'hit:6055_conv:12111', 'hit:5830_conv:11660', 'hit:6311_conv:12622', 'hit:6564_conv:13129',
    'hit:525_conv:1051', 'hit:903_conv:1806', 'hit:2705_conv:5410', 'hit:5406_conv:10812'
]

extended_comparing_responses_sample = extended_comparing_responses_sample[~extended_comparing_responses_sample['conv_id'].isin(conv_ids_to_drop)]

# Save the extended sample to a CSV file
extended_comparing_responses_sample.to_csv('comparing_responses_sample.csv', index=False)

#DO NOT TOUCH OR DELETE THE SAMPLE ANYMORE!!!!