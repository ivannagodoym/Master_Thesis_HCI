import pandas as pd
import numpy as np
import openai

# Set pandas display options to show all rows and columns
pd.set_option('display.max_colwidth', None)

#Data Processing:
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
data = pd.read_csv(f"{data_path}/thesis_data.csv", sep=",",on_bad_lines='skip')
thesis_df = pd.DataFrame(data)

thesis_df_reader = pd.read_csv(f"{data_path}/thesis_data.csv", sep=",", chunksize=18, iterator=True)
#test_sample = thesis_df.head(15)

for chunk in thesis_df_reader:
    print(chunk)
    break  # Stop after printing the first chunk

openai.api_key = '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'

emotion_labels = ["surprised", "excited", "angry", "proud", "sad", "annoyed",
            "grateful", "lonely", "afraid", "terrified", "guilty", "impressed",
            "disgusted", "hopeful", "confident", "furious", "anxious", "anticipating",
            "joyful", "nostalgic", "disappointed", "prepared", "jealous", "content",
            "devastated", "sentimental", "embarrassed", "caring", "trusting", "ashamed",
            "apprehensive", "faithful"]


# Grouping by conv_id
conv_sample = thesis_df.groupby('conv_id')['utterance'].apply(list).reset_index().sort_values(by='conv_id')  # sort_index(). #reset_index(name='utterance')
#test_conv_sample = conv_sample.head(15)

# Extract unique conversation IDs from grouped_data
conversation_ids = conv_sample['conv_id'].tolist()

# Extract emotion labels corresponding to the conversation IDs
emotions_list = thesis_df.groupby('conv_id')['emotion_label'].unique().tolist()
emotions_list = np.concatenate(emotions_list).tolist() #Flatten the list of arrays using numpy.concatenate() and Convert the flattened numpy array to a list
#test_emotions_sample = emotions_list[:15]


#function to generate empathetic responses based on situation, emotion, and the first utterance
def generate_empathetic_response(context, emotion, first_utterance):
    prompt = f"Consider a situation and the emotion of a speaker. Situation: {context}\nEmotion: {emotion}\nAct as the listener and reply to the speaker.\nSpeaker: {first_utterance}\nListener:"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt}],
        temperature=0.7,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    #print(prompt)
    return response.choices[0].message.content

# Create a new dataframe to store the listener responses
test21_responses = pd.DataFrame(columns=['conv_id', 'Test2.1_response'])

for index, row in conv_sample.iterrows(): #test_conv_sample.iterrows():
    conv_id = row['conv_id']
    conversation_context = thesis_df.loc[thesis_df['conv_id'] == conv_id, 'context'].iloc[0]

    # Extract emotion labels for the conversation
    emotions = thesis_df.loc[thesis_df['conv_id'] == conv_id, 'emotion_label'].tolist()

    # Selecting the most frequent emotion label for the conversation
    emotion = max(set(emotions), key=emotions.count)

    # Extract first utterance for the empathetic response
    first_utterance = row['utterance'][0]

    # Generate empathetic response
    empathetic_response = generate_empathetic_response(conversation_context, emotion, first_utterance)
    #print("Empathetic response:", empathetic_response)

    # Save the conversation ID and listener response in the dataframe
    new_row = pd.DataFrame({
        'conv_id': [conv_id],
        'Test2.1_response': [empathetic_response]
    })
    #Append the new row to the test21_responses DataFrame
    test21_responses = pd.concat([test21_responses, new_row], ignore_index=True)

# Save the dataframe to a CSV file
test21_responses.to_csv("test21_responses.csv", index=False)
