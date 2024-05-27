import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, classification_report


#read the data
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"

test1_data = pd.read_csv(f"{data_path}/EmotionPrediction_GPT35_nodup.csv", sep=",",on_bad_lines='skip')

pd.set_option('display.max_columns', None) #show all columns
#print(test1_data.head(10))

# Extract emotions in positions 2 and 3
test1_data['emotion_2'] = test1_data['top_3_emotions_responses'].str.split(',').str[1].str.strip().str.strip("[]'")
test1_data['emotion_3'] = test1_data['top_3_emotions_responses'].str.split(',').str[2].str.strip().str.strip("[]'")

#change names of columns:
test1_data.rename(columns = {'top_1_emotion_responses': '1st_Emotion_Pred', 'emotion_2': '2nd_Emotion_Pred',
                             'emotion_3': '3rd_Emotion_Pred', 'top_3_emotions_responses': 'Top_3_Emotion_Pred',
                             'original_labels': 'Original_Labels', 'position_of_original_label': 'Pred_Emo_Position'},
                            inplace = True)
#reorder
test1_data = test1_data[
    ['conv_id', 'Original_Labels', 'Top_3_Emotion_Pred', '1st_Emotion_Pred', '2nd_Emotion_Pred', '3rd_Emotion_Pred',
     'Pred_Emo_Position']]

test1_data['1st_Emotion_Pred'] = test1_data['1st_Emotion_Pred'].str.strip()
test1_data['2nd_Emotion_Pred'] = test1_data['2nd_Emotion_Pred'].str.strip()
test1_data['3rd_Emotion_Pred'] = test1_data['3rd_Emotion_Pred'].str.strip()

test1_data['1st_Emotion_Pred'] = test1_data['1st_Emotion_Pred'].str.lower()
test1_data['2nd_Emotion_Pred'] = test1_data['2nd_Emotion_Pred'].str.lower()
test1_data['3rd_Emotion_Pred'] = test1_data['3rd_Emotion_Pred'].str.lower()

# List of emotions
emotion_labels = ["surprised", "excited", "angry", "proud", "sad", "annoyed",
            "grateful", "lonely", "afraid", "terrified", "guilty", "impressed",
            "disgusted", "hopeful", "confident", "furious", "anxious", "anticipating",
            "joyful", "nostalgic", "disappointed", "prepared", "jealous", "content",
            "devastated", "sentimental", "embarrassed", "caring", "trusting", "ashamed",
            "apprehensive", "faithful"]

# Display the updated DataFrame
print(test1_data.head())

#print(test1_data.describe())ç
accuracy_df = pd.DataFrame(test1_data)

#DATA ANALYSIS
accuracy_top1 = round(accuracy_score(test1_data['Original_Labels'], test1_data['1st_Emotion_Pred']), 2)
accuracy_top2 = round(accuracy_score(test1_data['Original_Labels'], test1_data['2nd_Emotion_Pred']), 2)
accuracy_top3 = round(accuracy_score(test1_data['Original_Labels'], test1_data['3rd_Emotion_Pred']), 2)

# Function to check if original label is present in top 3 predicted emotions
def is_label_in_top3(row):
    top_3_emotions = eval(row['Top_3_Emotion_Pred'])  # Convert string to list
    return row['Original_Labels'] in top_3_emotions if top_3_emotions else False  # Handle None values
# Create a new column indicating whether original label is present in top 3
test1_data['is_label_in_top3'] = test1_data.apply(is_label_in_top3, axis=1)
# Calculate accuracy considering only cases where original label is present in top 3
overall_accuracy = round(accuracy_score(test1_data['is_label_in_top3'], test1_data['Pred_Emo_Position'].notnull()), 2)

# Create accuracy table
accuracy_table = pd.DataFrame({
    'Position': ['Top1_Emotion_Prediction', 'Top2_Emotion_Prediction', 'Top3_Emotion_Prediction', 'Overall_Accuracy'],
    'Accuracy': [f'{accuracy_top1}', f'{accuracy_top2}', f'{accuracy_top3}', f'{overall_accuracy}']
})
print(accuracy_table)

# Count occurrences where the original labeled emotion is present and is top 1
top_1_count = test1_data['Pred_Emo_Position'].eq(1).sum()

# Count occurrences where the original labeled emotion is present in top 3
top_3_count = test1_data['Pred_Emo_Position'].notnull().sum()

# Calculate percentages
total_rows = len(test1_data)
percentage_top_3 = (top_3_count / total_rows) * 100
percentage_top_1 = (top_1_count / total_rows) * 100
print("Total rows:", total_rows)
print("Top 1 count:", top_1_count)
print("Top 3 count:", top_3_count)
print("Percentage of times original labeled emotion appeared in top 3 responses:", "{:.2f}%".format(percentage_top_3))
print("Percentage of times original labeled emotion appeared as top 1 response:", "{:.2f}%".format(percentage_top_1))


predicted_labels = test1_data['1st_Emotion_Pred'].tolist()
actual_labels = test1_data['Original_Labels'].tolist()

predicted_series = pd.Series(predicted_labels)
actual_series = pd.Series(actual_labels)

predicted_counts = predicted_series.value_counts()
actual_counts = actual_series.value_counts()

data = pd.DataFrame({'Predicted': predicted_counts, 'Actual': actual_counts})
data = data.reset_index().melt(id_vars='index', var_name='Model', value_name='Count')
data = data.sort_values(by='Model', ascending=False)
data = data.sort_values(by='index')

# Plotting
plt.figure(figsize=(16, 10))
sns.barplot(x='index', y='Count', hue='Model', data=data, palette=['#BAA4CA', '#028090'])
plt.title('Predicted vs Actual Labels')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

#savingg
#plt.savefig('PredictedVsActualLabels_Test1_ha.png')
#plt.show()

#Check hallucinated data distribution
#load hallucinated only data:
hallucinated_df = pd.read_csv(f"{data_path}/EmotionPrediction_GPT35_hallucinated.csv", sep=",",on_bad_lines='skip')
hallucinated_df.describe()

predicted_hallucinated = hallucinated_df['1st_Emotion_Pred'].tolist()
predicted_series = pd.Series(predicted_labels)
predicted_counts = predicted_series.value_counts()

# Plotting verrückness
plt.figure(figsize=(16, 10))
predicted_counts.plot(kind='bar', color='#028090')
plt.title('Distribution of 1st Predicted Emotions in Hallucinated Data')
plt.xlabel('Predicted Emotions')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Hallucinated_Distribution_Test1.png')
plt.show()


#Getting the whole list of emotions:
all_emotions = pd.concat([test1_data['1st_Emotion_Pred'], test1_data['2nd_Emotion_Pred'], test1_data['3rd_Emotion_Pred']])
unique_labels = all_emotions.unique()
hallucinated_labels = set(unique_labels) - set(emotion_labels)
hallucinated_labels = list(hallucinated_labels)
print("Hallucinated labels:", hallucinated_labels)

list_unique_hallucinations = [
    'worthless', 'vengeful', 'forgetful', 'secretive', 'unsettled', 'hopeless', 'avoiding', 'funny', 'passionate',
    'motivated', 'cared for', 'supporting', 'addicted', 'intrigued', 'understanding', 'encouraging', 'agreeing',
    'joking', 'glad', 'frightened', 'nauseated', 'bothered', 'indecisive', 'pessimistic', 'lazy', 'alone', 'patriotic',
    'thrilled', 'upset', 'skeptical', 'appalled', 'grieving', 'depressed', 'hungry', 'trustin', 'regretful', 'undecided',
    'stressed', 'empathetic', 'ambitious', 'comforting', 'hurt', 'courageous', 'pity', 'suspicious',
    'gross', 'uneasy', 'dislike', 'contemplating', 'bored', 'desensitized', 'calm', 'confused', 'impatient', 'expecting',
    'peaceful', 'protective', 'curious', 'sarcastic', 'distrustful', 'uncomfortable', 'reassured', 'frustrated', 'painful',
    'repulsed', 'aggravated', 'silly', 'forgiving', 'rewarding', 'impressive', 'inspired', 'hate', 'appreciated', 'committed',
    'indifferent', 'compassionate', 'longing', 'helpless', 'bewildered', 'calming', 'yearning', 'not surprised', 'optimistic',
    'inferior', 'amused', 'envious', 'inquiring', 'resigned', 'aprehensive', 'trustingspeaker', 'relief', 'trust', 'creepy',
    'tired', 'encouraged', 'doubtful', 'understood', 'sorry', 'misunderstood', 'conflicted', 'adventurous', 'defiant', 'concerned',
    'prepared', 'comfortable', 'pleased', 'humiliated', 'irritated', 'admiration', 'appreciating', 'guilty', 'shocked', 'nervous',
    'stressful', 'advised', 'relieved', 'hesitant', 'fond', 'helpful', 'disinterested', 'happy', 'mortified', 'remorseful', 'missing',
    'unsure', 'patient', 'worried', 'betrayed', 'relatable', 'uncertain', 'desperate', 'humorous', 'reflecting', 'loyal', 'paranoid',
    'admirable', 'accomplished', 'safe', 'thankful', 'sympathetic', 'cautious', 'interested', 'apologetic', 'relaxed', 'loving',
    'overwhelmed', 'grossed out', 'reluctant', 'emotional', 'pain', 'insecure', 'agreed', 'satisfied', 'touched', 'supportive',
    'joyul', 'attached', 'tempted', 'neutral', 'lucky', 'supported', 'exhausted', 'elated', 'sick', 'intimidated', 'determined',
    'trustful', 'horrified', 'brave', 'trustingsentimental', 'unsurprised', 'heartbroken', 'admiring', 'advice', 'loved', 'amazed',
    'horrifying', 'cranky', 'unconcerned', 'offended', 'questioning', 'puzzled', 'competitive', 'trustworthy', 'comforted', 'fearful',
    'wistful', 'awkward', 'scared'
]

#count hallucinated labels
print(len(list_unique_hallucinations))

#overall distribution of verrückness
# Count occurrences of each unique hallucinated label
hallucinated_counts = {label: 0 for label in list_unique_hallucinations}
for column in ['1st_Emotion_Pred']:
    for label in hallucinated_df[column]:
        if label in hallucinated_counts:
            hallucinated_counts[label] += 1

# Plotting
plt.figure(figsize=(25, 8))
plt.bar(hallucinated_counts.keys(), hallucinated_counts.values(), color='skyblue')
plt.title('Distribution of Hallucination First Emotion')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

total_hallucinations = sum(hallucinated_counts.values())
print("Total Hallucinations:", total_hallucinations)


