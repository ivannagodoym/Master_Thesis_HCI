import pandas as pd
import numpy as np
import openai

# Set pandas display options to show all rows and columns
pd.set_option('display.max_colwidth', None)

#Data Processing:
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"

data = pd.read_csv(f"{data_path}/thesis_data.csv", sep=",",on_bad_lines='skip')
thesis_df = pd.DataFrame(data)

# Grouping by conv_id
conv_sample = thesis_df.groupby('conv_id')['utterance'].apply(list).reset_index().sort_values(by='conv_id')  # sort_index(). #reset_index(name='utterance')
test_conv_sample = conv_sample.head(20)

# Extract unique conversation IDs from grouped_data
conversation_ids = conv_sample['conv_id'].tolist()
test_conv_ids = conversation_ids[:20]

# Extract emotion labels corresponding to the conversation IDs
emotions_list = thesis_df.groupby('conv_id')['emotion_label'].unique().tolist()
emotions_list = np.concatenate(emotions_list).tolist() #Flatten the list of arrays using numpy.concatenate() and Convert the flattened numpy array to a list
test_emotions_sample = emotions_list[:20]

emotion_labels = ["surprised", "excited", "angry", "proud", "sad", "annoyed",
            "grateful", "lonely", "afraid", "terrified", "guilty", "impressed",
            "disgusted", "hopeful", "confident", "furious", "anxious", "anticipating",
            "joyful", "nostalgic", "disappointed", "prepared", "jealous", "content",
            "devastated", "sentimental", "embarrassed", "caring", "trusting", "ashamed",
            "apprehensive", "faithful"]

openai.api_key = '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'

# Initialize lists to store data
total_conversations = len(test_conv_sample) #len(conv_sample)
speaker_role = "Speaker"
listener_role = "Listener"

original_labels = []
top_3_emotions_responses = []
top_1_emotion_responses = []
position_of_original_label = []
conversation_ids_list = []

# Function to process conversations
def process_conversations(conversations, emotion_labels):
    original_label_in_top_3_count = 0
    original_label_in_top_1_count = 0

    # Loop through each conversation
    for index, row in conversations.iterrows(): #conv_sample.iterrows(): test_conv_sample
        conversation_id = row['conv_id']
        conversation = row['utterance']
        original_label = emotions_list[index]  # Fetching corresponding emotion label for this conversation

        # Store original label
        original_labels.append(original_label)
        conversation_ids_list.append(conversation_id)

        # Initialize variables to store conversation lines and emotions
        conversation_lines = []
        prompt = ""

        prompt = f"A conversation between a Speaker and a Listener will be given. The conversation contains several utterances clearly divided, the Speaker always speaks first and the Listener replies. Choose the top 3 emotions from the list that best represents the emotions of the speaker. Always start with the most predominant emotion.\n"
        prompt += "The list of emotions is: " + ", ".join(emotion_labels) + "\n\n"

        # Iterate through each utterance in the conversation
        for i, utterance in enumerate(conversation):
            # Assign speaker and listener roles based on the utterance index
            role = speaker_role if i % 2 == 0 else listener_role

            # Append role and utterance to conversation lines
            conversation_lines.append(f"{role}: {utterance}")

            # Check if it's the speaker's turn to speak and add the utterance to the prompt
            prompt += f"{role}:\n{utterance}\n"

        # Call the OpenAI API
        response = openai.chat.completions.create(
            model="gpt-4", #gpt-4 turbo it does whatever.
            messages=[
                {
                    "role": "system",
                    "content": prompt
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Extract top emotions
        top_3_emotions = response.choices[0].message.content.strip().split(",")[:3]
        top_1_emotion = top_3_emotions[0]

        # Store API responses for top 3 and top 1 emotions
        top_3_emotions_responses.append(top_3_emotions)
        top_1_emotion_responses.append(top_1_emotion)

        # Determine position of original label among predicted labels
        position = None
        for i, emotion in enumerate(top_3_emotions):
            if original_label.strip().lower() == emotion.strip().lower():  # Compare original label and predicted emotion (case-insensitive)
                position = i + 1
                break
        position_of_original_label.append(position)

        # Calculate percentages
        original_label_in_top_3_count = sum(
            1 for pos in position_of_original_label if pos is not None)  # Count non-None positions
        original_label_in_top_1_count = sum(
            1 for pos in position_of_original_label if pos == 1)  # Count positions where original label is in 1st position

    return original_label_in_top_3_count, original_label_in_top_1_count

# Process the conversations
original_label_in_top_3_count, original_label_in_top_1_count = process_conversations(test_conv_sample, test_emotions_sample)

# Calculate percentages
total_conversations_processed = len(original_labels)
original_label_in_top_3_percentage = (original_label_in_top_3_count / total_conversations_processed) * 100
original_label_in_top_1_percentage = (original_label_in_top_1_count / total_conversations_processed) * 100

# Create a DataFrame to store the results
df_results = pd.DataFrame({
    "conv_id": conversation_ids_list,
    "original_labels": original_labels,
    "top_3_emotions_responses": top_3_emotions_responses,
    "top_1_emotion_responses": top_1_emotion_responses,
    "position_of_original_label": position_of_original_label
})

# Save results to CSV
df_results.to_csv("emotion_predictions.csv", index=False)

# Print results
print(f"Original label appeared in top 3 predictions in {original_label_in_top_3_percentage}% of conversations.")
print(f"Original label appeared as top prediction in {original_label_in_top_1_percentage}% of conversations.")
print(f"Original label appeared in top 3 predictions in {original_label_in_top_3_count} out of {total_conversations_processed} conversations.")
