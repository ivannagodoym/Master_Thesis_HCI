import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


'''
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
'''
#show all clolumns
pd.set_option('display.max_columns', None)

''' 
# Load the comparing_responses_sample.csv file
comparing_responses_sample = pd.read_csv('comparing_responses_sample.csv')

# Create new column for human evaluation
#Empathy comparison
#2.1 = 0
#2.2 = 1

#Eliminar de data: 'hit:3233_conv:6466'

#comparing_responses_sample['human_evaluation'] = 0

evaluation_results = pd.DataFrame()
conv_id = comparing_responses_sample['conv_id']
evaluation_results['conv_id'] = conv_id

human_evaluation_values_dict = {
    'hit:2541_conv:5083': 0,
    'hit:8167_conv:16335': 0,
    'hit:2948_conv:5897': 1,
    'hit:3336_conv:6673': 1,
    'hit:10880_conv:21761': 1,
    'hit:6026_conv:12053': 1,
    'hit:8145_conv:16290': 0,
    'hit:5248_conv:10497': 0,
    'hit:4389_conv:8778': 0,
    'hit:2218_conv:4437': 0,
    'hit:12281_conv:24562': 0,
    'hit:3204_conv:6408': 0,
    'hit:7683_conv:15366': 0,
    'hit:11539_conv:23079': 0,
    'hit:8645_conv:17291': 1,
    'hit:7920_conv:15840': 0,
    'hit:3958_conv:7916': 1,
    'hit:9960_conv:19920': 0,
    'hit:1790_conv:3581': 1,
    'hit:10165_conv:20330': 1,
    'hit:7741_conv:15482': 1,
    'hit:8889_conv:17778': 0,
    'hit:10656_conv:21313': 1,
    'hit:2954_conv:5908': 1,
    'hit:2648_conv:5296': 1,
    'hit:8326_conv:16653': 0,
    'hit:4392_conv:8785': 0,
    'hit:11086_conv:22172': 0,
    'hit:3951_conv:7903': 0,
    'hit:4113_conv:8227': 1,
    'hit:8608_conv:17217': 1,
    'hit:1822_conv:3644': 0,
    'hit:6624_conv:13248': 1,
    'hit:10742_conv:21485': 1,
    'hit:1696_conv:3393': 1,
    'hit:10492_conv:20984': 0,
    'hit:1652_conv:3305': 1,
    'hit:6241_conv:12483': 1,
    'hit:8966_conv:17932': 1,
    'hit:2812_conv:5625': 0,
    'hit:8176_conv:16352': 1,
    'hit:5978_conv:11956': 1,
    'hit:2530_conv:5061': 0,
    'hit:7007_conv:14015': 1,
    'hit:5274_conv:10549': 0,
    'hit:8502_conv:17004': 1,
    'hit:5848_conv:11697': 0,
    'hit:6632_conv:13264': 1,
    'hit:6267_conv:12535': 0,
    'hit:7503_conv:15006': 1,
    'hit:7161_conv:14322': 1,
    'hit:12154_conv:24309': 0,
    'hit:417_conv:834': 1,
    'hit:10850_conv:21701': 0,
    'hit:5610_conv:11221': 1,
    'hit:658_conv:1317': 0,
    'hit:1624_conv:3248': 0,
    'hit:8507_conv:17014': 0,
    'hit:8480_conv:16960': 0,
    'hit:4770_conv:9541': 1,
    'hit:3570_conv:7141': 1,
    'hit:3335_conv:6670': 1,
    'hit:6156_conv:12313': 1,
    'hit:9359_conv:18719': 1,
    'hit:9204_conv:18409': 0,
    'hit:8358_conv:16717': 1,
    'hit:8257_conv:16514': 0,
    'hit:7923_conv:15847': 1,
    'hit:8092_conv:16184': 0,
    'hit:6541_conv:13082': 1,
    'hit:7930_conv:15861': 0,
    'hit:10825_conv:21651': 0,
    'hit:6592_conv:13184': 0,
    'hit:7074_conv:14148': 0,
    'hit:9454_conv:18908': 0,
    'hit:8148_conv:16297': 1,
    'hit:7144_conv:14289': 0,
    'hit:8676_conv:17352': 1,
    'hit:9415_conv:18830': 1,
    'hit:3351_conv:6702': 0,
    'hit:5127_conv:10255': 0,
    'hit:8483_conv:16966': 1,
    'hit:9612_conv:19225': 1,
    'hit:5281_conv:10562': 0,
    'hit:2684_conv:5369': 1,
    'hit:7548_conv:15096': 1,
    'hit:5385_conv:10770': 0,
    'hit:2964_conv:5929': 0,
    'hit:9032_conv:18064': 1,
    'hit:1151_conv:2303': 0,
    'hit:2899_conv:5799': 1,
    'hit:9611_conv:19222': 0,
    'hit:11778_conv:23556': 0,
    'hit:8324_conv:16648': 1,
    'hit:9787_conv:19575': 0,
    'hit:7425_conv:14850': 1,
    'hit:3920_conv:7840': 1,
    'hit:9720_conv:19441': 1,
    'hit:4038_conv:8077': 0,
    'hit:7116_conv:14233': 0
}

# Create a list to store the manually entered values based on 'conv_id'
evaluation_values = [human_evaluation_values_dict.get(conv_id, None) for conv_id in evaluation_results['conv_id']]

# Assign the list of evaluation values to the 'evaluation_value' column in evaluation_results
evaluation_results['evaluation_value'] = evaluation_values

# Add the evaluation_values column to comparing_responses_sample DataFrame
comparing_responses_sample['evaluation_value'] = evaluation_results['evaluation_value']

print(comparing_responses_sample)

#drop column hit:3233_conv:6466
comparing_responses_sample = comparing_responses_sample[comparing_responses_sample.conv_id != 'hit:3233_conv:6466']

# Save the comparing_responses_sample DataFrame to a CSV file
comparing_responses_sample.to_csv('comparing_responses_sample.csv', index=False)
'''

#ANALYSIS
#load the comparing_responses_sample.csv file
comparing_responses_sample = pd.read_csv('comparing_responses_sample.csv')
comparing_responses_sample.describe()

# Group data by evaluation value and calculate counts for each group
group_counts = comparing_responses_sample['evaluation_value'].value_counts()
print(group_counts)

# Plot the distribution of evaluation values
plt.figure(figsize=(8, 6))
plt.bar(group_counts.index, group_counts.values, color='#028090', alpha=0.8, width=0.3, align='center')
plt.ylabel('Count')
plt.title('Test 2 Human Evaluation Subset')
plt.xticks([0, 1], labels=['Emotion Background (Test 2.1)', 'No Background (Test 2.2)'])
plt.savefig('Test2_HumanEvaluation.png')
plt.show()

# Calculate the mean and standard deviation of the evaluation values
mean_evaluation = comparing_responses_sample['evaluation_value'].mean()
std_evaluation = comparing_responses_sample['evaluation_value'].std()
print(f'Mean Evaluation Value: {mean_evaluation}')
print(f'Standard Deviation of Evaluation Values: {std_evaluation}')



