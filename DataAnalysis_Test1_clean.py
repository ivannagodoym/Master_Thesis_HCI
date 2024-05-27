import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, classification_report


#Processing
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
test1_data = pd.read_csv(f"{data_path}/EmotionPrediction_GPT35_clean.csv", sep=",",on_bad_lines='skip')
pd.set_option('display.max_columns', None) #show all columns

''' # Extract emotions in positions 2 and 3
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

# Filter DataFrame to keep only rows where the emotions in the prediction columns are in the emotion_labels list
filtered_df = test1_data[test1_data['1st_Emotion_Pred'].isin(emotion_labels) &
                 test1_data['2nd_Emotion_Pred'].isin(emotion_labels) &
                 test1_data['3rd_Emotion_Pred'].isin(emotion_labels)]

# DataFrame containing filtered-out data
filtered_out_df = test1_data[~test1_data.index.isin(filtered_df.index)]

#save to csv
filtered_df.to_csv('EmotionPrediction_GPT35_clean.csv', index=False)
filtered_out_df.to_csv('EmotionPrediction_GPT35_hallucinated.csv', index=False)
'''

#DATA ANALYSIS

#ACCURACY PER POSITION
accuracy_top1 = round(accuracy_score(test1_data['Original_Labels'], test1_data['1st_Emotion_Pred']), 2)
accuracy_top2 = round(accuracy_score(test1_data['Original_Labels'], test1_data['2nd_Emotion_Pred']), 2)
accuracy_top3 = round(accuracy_score(test1_data['Original_Labels'], test1_data['3rd_Emotion_Pred']), 2)

# Function to check if original label is present in top 3 predicted emotions
def is_label_in_top3(row):
    top_3_emotions = eval(row['Top_3_Emotion_Pred'])
    return row['Original_Labels'] in top_3_emotions if top_3_emotions else False  # Handle None values

# Create a new column indicating whether original label is present in top 3
test1_data['is_label_in_top3'] = test1_data.apply(is_label_in_top3, axis=1)
# Calculate accuracy considering only cases where original label is present in top 3
overall_accuracy = round(accuracy_score(test1_data['is_label_in_top3'], test1_data['Pred_Emo_Position'].notnull()), 3)
#Overall accuracy is when it predicted the emotion in the top 3 selection.
# Create accuracy table
accuracy_table = pd.DataFrame({
    'Position': ['Prediction as 1st Emotion', 'Prediction as 2nd Emotion', 'Prediction as 3rd Emotion', 'Prediction in Top 3 Emotions'],
    'Accuracy': [f'{accuracy_top1}', f'{accuracy_top2}', f'{accuracy_top3}', f'{overall_accuracy}']
})

accuracy_table = accuracy_table.round(2)
#save to csv
accuracy_table.to_csv('TablePositionsAccuracy_Test1.csv', index=False)
print(accuracy_table)

''' 
#Percentage just in case we want to show it
# Count occurrences where the original labeled emotion is present and is top 1
top_1_count = test1_data['Pred_Emo_Position'].eq(1).sum()
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
'''

# List of emotions
emotion_labels = ["surprised", "excited", "angry", "proud", "sad", "annoyed",
            "grateful", "lonely", "afraid", "terrified", "guilty", "impressed",
            "disgusted", "hopeful", "confident", "furious", "anxious", "anticipating",
            "joyful", "nostalgic", "disappointed", "prepared", "jealous", "content",
            "devastated", "sentimental", "embarrassed", "caring", "trusting", "ashamed",
            "apprehensive", "faithful"]


'''
#Precision for top 3 emotions
top_3_emotions_responses = test1_data['Top_3_Emotion_Pred'].apply(eval).tolist()
y_true_all = test1_data['Original_Labels'].tolist()
unique_emotions = set(y_true_all) # Get unique emotion classes

# Calculate precision for the top three predicted emotions
for emotion in unique_emotions:
    # Convert labels to binary for the current emotion class
    y_true_binary = [1 if label == emotion else 0 for label in y_true_all]

    # Convert top 3 predicted emotions to binary
    y_pred_binary = []
    for top_3_emotions in top_3_emotions_responses:
        if emotion in top_3_emotions:
            y_pred_binary.append(1)
        else:
            y_pred_binary.append(0)

    # Calculate precision for the current emotion class
    precision = precision_score(y_true_binary, y_pred_binary)

    print(f"Precision for '{emotion}' among top 3 predicted emotions: {precision}")

#Precision overall 3emotions (cannot be done right now)


#mAP:
y_true_all = test1_data['Original_Labels'].tolist()  # True labels for all instances
y_pred_all = test1_data['Top_3_Emotion_Pred'].apply(eval).tolist()  # Predicted labels for all instances

# Assign probabilities based on the position of the predicted emotion
def assign_probabilities(predictions):
    num_predictions = len(predictions)
    probabilities = {}
    for i, emotion in enumerate(predictions):
        probabilities[emotion] = 1 - (i / num_predictions)  # Assign higher probability to higher ranking positions
    return probabilities

# Calculate probabilities for all instances
output_probabilities = [assign_probabilities(predictions) for predictions in y_pred_all]

# Initialize lists to store AP values for each emotion
ap_values = []
# Calculate average precision (AP) for each emotion
for i, emotion in enumerate(unique_emotions):
    # Convert labels to binary for the current emotion class
    y_true_binary = [1 if emotion in labels else 0 for labels in y_true_all]
    # Extract predicted probabilities for current emotion
    y_scores_binary = [output_probabilities[j][emotion] if emotion in output_probabilities[j] else 0 for j in
                       range(len(y_pred_all))]
    # Calculate average precision (AP) for the current emotion class
    ap = average_precision_score(y_true_binary, y_scores_binary)
    # Append AP value to list
    ap_values.append(ap)

# Calculate mAP
mAP = np.mean(ap_values)
print("Mean Average Precision (mAP):", mAP)
'''

#Confusion Matrix
labels = sorted(set(test1_data['Original_Labels']) | set(test1_data['1st_Emotion_Pred']))
label_to_int = {label: i for i, label in enumerate(labels)}
int_to_label = {i: label for label, i in label_to_int.items()}

y_true = np.array([label_to_int[label] for label in test1_data['Original_Labels']])
y_pred = np.array([label_to_int[label] for label in test1_data['1st_Emotion_Pred']])

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)


#cmap=sns.color_palette("light:#5A9", as_cmap=True) #color palette
#Plot sbn heatmap
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True), annot_kws={"fontsize": 7.5}, xticklabels=labels, yticklabels=labels, linewidths=1)
plt.xlabel('Top Predicted Label')
plt.ylabel('Original Label')
plt.title('Confusion Matrix')
#plt.savefig('ConfusionMatrix_Test1.png')

plt.show()



# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate micro-average precision, recall, and F1-score
#micro_precision = precision_score(y_true, y_pred, average='micro')
#micro_recall = recall_score(y_true, y_pred, average='micro')
#micro_f1 = f1_score(y_true, y_pred, average='micro')

# Calculate macro-average precision, recall, and F1-score
macro_precision = precision_score(y_true, y_pred, average='macro')
macro_recall = recall_score(y_true, y_pred, average='macro')
macro_f1 = f1_score(y_true, y_pred, average='macro')

# Calculate weighted-average precision, recall, and F1-score
weighted_precision = precision_score(y_true, y_pred, average='weighted')
weighted_recall = recall_score(y_true, y_pred, average='weighted')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')

#Make table with rows: accuracy, micro, macro, weighted
extra_metrics_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Macro-Average', 'Weighted-Average'],
    'Precision': [accuracy, macro_precision, weighted_precision],
    'Recall': [accuracy, macro_recall, weighted_recall],
    'F1-Score': [accuracy, macro_f1, weighted_f1]
})
#round to 2 decimals
extra_metrics_table = extra_metrics_table.round(2)
#save to csv
extra_metrics_table.to_csv('TableMetricsSmall_Test1.csv', index=False)

print(extra_metrics_table)

# Compute the classification report
cls_report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

# Extract metrics from the report
metrics = {label: cls_report[label] for label in labels}
# Convert metrics to DataFrame for easy plotting
metrics_df = pd.DataFrame.from_dict(metrics, orient='index')

#plot heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(metrics_df[['precision', 'recall', 'f1-score']], annot=True, fmt='.2f', cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))
plt.title('Classification Metrics Top Predicted Emotion')
plt.xlabel('Metric')
plt.ylabel('Emotion')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.savefig('ClassificationMetrics_Test1.png')

plt.show()

#add extra metrics to the metrics_df table
total_support = sum(cls_report[label]['support'] for label in cls_report if label not in ['accuracy', 'macro avg', 'weighted avg'])

metrics_df.loc['accuracy'] = accuracy, accuracy, accuracy, total_support
#metrics_df.loc['micro avg'] = micro_precision, micro_recall, micro_f1, total_support
metrics_df.loc['macro avg'] = macro_precision, macro_recall, macro_f1, total_support
metrics_df.loc['weighted avg'] = weighted_precision, weighted_recall, weighted_f1, total_support

metrics_rounded = metrics_df.round(2)
metrics_rounded['support'] = metrics_rounded['support'].astype(int)

print(metrics_rounded)
#save to csv
metrics_rounded.to_csv('Table_ClassificationMetricsTest1.csv', index=True)

predicted_labels = test1_data['1st_Emotion_Pred'].tolist()
actual_labels = test1_data['Original_Labels'].tolist()

predicted_series = pd.Series(predicted_labels)
actual_series = pd.Series(actual_labels)

predicted_counts = predicted_series.value_counts()
print(predicted_counts)
actual_counts = actual_series.value_counts()

data = pd.DataFrame({'Predicted': predicted_counts, 'Actual': actual_counts})
data = data.reset_index().melt(id_vars='index', var_name='Model', value_name='Count')
data = data.sort_values(by='Model', ascending=False)
data = data.sort_values(by='index')

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='index', y='Count', hue='Model', data=data, palette=['#BAA4CA', '#028090'])
plt.title('Predicted vs Actual Labels')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

#savingg
plt.savefig('PredictedVsActualLabels_Test1.png')

plt.show()