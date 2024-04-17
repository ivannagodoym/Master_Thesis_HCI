import pandas as pd
import numpy as np
import openai

# Set pandas display options to show all rows and columns
pd.set_option('display.max_colwidth', None)

# Data Processing:
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
chunk_size = 30  # Define the size of each chunk
thesis_df_reader = pd.read_csv(f"{data_path}/thesis_data.csv", sep=",", chunksize=chunk_size, iterator=True)
print(thesis_df_reader)

# Iterate over the first chunk and print it
#for chunk in thesis_df_reader:
#    print(chunk)
#    break  # Stop after printing the first chunk


openai.api_key = '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'


emotion_labels = ["surprised", "excited", "angry", "proud", "sad", "annoyed",
            "grateful", "lonely", "afraid", "terrified", "guilty", "impressed",
            "disgusted", "hopeful", "confident", "furious", "anxious", "anticipating",
            "joyful", "nostalgic", "disappointed", "prepared", "jealous", "content",
            "devastated", "sentimental", "embarrassed", "caring", "trusting", "ashamed",
            "apprehensive", "faithful"]

# Initialize lists to store data
speaker_role = "Speaker"
listener_role = "Listener"

original_labels = []
top_3_emotions_responses = []
top_1_emotion_responses = []
position_of_original_label = []
conversation_ids_list = []
emotions_list = []

total_conversations_processed = 0

# Function to process data chunk
def process_data_chunk(chunky):
    # Grouping by conv_id
    conv_sample = chunky.groupby('conv_id')['utterance'].apply(list).reset_index().sort_values(by='conv_id')
    # Extract unique conversation IDs from grouped_data
    conversation_ids = conv_sample['conv_id'].tolist()

    # Extract emotion labels corresponding to the conversation IDs
    emotions_list = chunky.groupby('conv_id')['emotion_label'].unique().tolist()
    emotions_list = np.concatenate(emotions_list).tolist()

    # Process the conversations
    original_label_in_top_3_count, original_label_in_top_1_count = process_conversations(conv_sample, emotion_labels, emotions_list) # Call the function to process conversations

    total_conversations_processed = len(conv_sample)

    return original_label_in_top_3_count, original_label_in_top_1_count, total_conversations_processed


# Function to process conversations
def process_conversations(conversations, emotion_labels, emotions_list):
    original_label_in_top_3_count = 0
    original_label_in_top_1_count = 0

    # Loop through each conversation
    for index, row in conversations.iterrows(): #conv_sample.iterrows(): test_conv_sample
        conversation_id = row['conv_id']
        conversation = row['utterance']
        # Fetching corresponding emotion label for this conversation
        original_label = emotions_list[index]

        # Store original label
        original_labels.append(original_label)
        conversation_ids_list.append(conversation_id)

        # Initialize variables to store conversation lines and emotions
        conversation_lines = []
        #prompt = ""
        prompt = f"A conversation between a {speaker_role} and a {listener_role} will be given. The conversation contains several utterances clearly divided, the {speaker_role} always speaks first and the {listener_role} replies.\n"
        prompt += f"Consider exclusively the following list of emotions for labeling: surprised, excited, angry, proud, sad, annoyed, grateful, lonely, afraid, terrified, guilty, impressed, disgusted, hopeful, confident, furious, anxious, anticipating, joyful, nostalgic, disappointed, prepared, jealous, content, devastated, sentimental, embarrassed, caring, trusting, ashamed, apprehensive, faithful\n"
        prompt += f"Always choose the top 3 emotions that best represents the emotions of the speaker and only use emotions from the given list. Always start with the most predominant emotion. Do not use an emotion that is not part of the list.\n"
        prompt += "Always output your answers in the following format exclusively: emotion,emotion,emotion\n"

        # Iterate through each utterance in the conversation
        for i, utterance in enumerate(conversation):
            # Assign speaker and listener roles based on the utterance index
            role = speaker_role if i % 2 == 0 else listener_role

            # Append role and utterance to conversation lines
            conversation_lines.append(f"{role}: {utterance}")

            # Check if it's the speaker's turn to speak and add the utterance to the prompt
            prompt += f"{role}:\n{utterance}\n"

        # Print the prompt
        #print("Prompt sent to OpenAI:")
        #print(prompt)

        # Call the OpenAI API
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "system",
                    "content": prompt
                }
            ],
            max_tokens=256, #100
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        # Extract top emotions
        top_3_emotions = response.choices[0].message.content.strip().split(",")[:3]
        top_1_emotion = top_3_emotions[0]

        # Store API responses for top 3 and top 1 emotions
        top_3_emotions_responses.append(top_3_emotions)
        top_1_emotion_responses.append(top_1_emotion)

        # Print the response from OpenAI
        #print("Response from OpenAI:")
        #print(response.choices[0].message.content)


        # Determine position of original label among predicted labels
        position = None
        for i, emotion in enumerate(top_3_emotions):
            if original_label.strip().lower() == emotion.strip().lower():  #Compare original label and predicted emotion (case-insensitive)
                position = i + 1
                break
        position_of_original_label.append(position)

        # Increment counters if original label is in top 3 or top 1
        if position is not None:
            original_label_in_top_3_count += 1
            if position == 1:
                original_label_in_top_1_count += 1

    return original_label_in_top_3_count, original_label_in_top_1_count

# Process a limited number of chunks for testing
num_chunks_to_process = 1434
total_chunks_processed = 0
total_original_label_in_top_3_count = 0
total_original_label_in_top_1_count = 0

for chunk_number, chunk in enumerate(thesis_df_reader):
    if chunk_number >= num_chunks_to_process:
        break

    chunk_original_label_in_top_3_count, chunk_original_label_in_top_1_count, chunk_total_conversations_processed = process_data_chunk(chunk)

    total_chunks_processed += 1
    print(f"Chunks processed:", {total_chunks_processed})
    total_conversations_processed += chunk_total_conversations_processed
    total_original_label_in_top_3_count += chunk_original_label_in_top_3_count
    total_original_label_in_top_1_count += chunk_original_label_in_top_1_count

# Calculate overall percentages
overall_original_label_in_top_3_percentage = (total_original_label_in_top_3_count / total_conversations_processed) * 100
overall_original_label_in_top_1_percentage = (total_original_label_in_top_1_count / total_conversations_processed) * 100

# Create a DataFrame to store the results
df_results = pd.DataFrame({
    "conv_id": conversation_ids_list,
    "original_labels": original_labels,
    "top_3_emotions_responses": top_3_emotions_responses,
    "top_1_emotion_responses": top_1_emotion_responses,
    "position_of_original_label": position_of_original_label
})

# Save results to CSV
df_results.to_csv("EmotionPrediction_GPT35.csv", index=False)



# Print results
print(f"Original label appeared in top 3 predictions in {overall_original_label_in_top_3_percentage}% of {total_conversations_processed} conversations.")
print(f"Original label appeared as top prediction in {overall_original_label_in_top_1_percentage}% of {total_conversations_processed} conversations.")

