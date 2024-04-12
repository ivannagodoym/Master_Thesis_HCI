
import pandas as pd
import numpy as np
import openai

# Set pandas display options to show all rows and columns
pd.set_option('display.max_colwidth', None)
#pd.set_option('display.width', None)

#Data Processing:
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
data = pd.read_csv(f"{data_path}/thesis_data.csv", sep=",",on_bad_lines='skip')
thesis_df = pd.DataFrame(data)
test_sample = thesis_df.head(15)
#print(thesis_df.head())

# Grouping by conv_id
conv_sample = thesis_df.groupby('conv_id')['utterance'].apply(list).reset_index().sort_values(by='conv_id')  # sort_index(). #reset_index(name='utterance')
test_conv_sample = conv_sample.head(15)
#print(test_conv_sample)

# Extract unique conversation IDs from grouped_data
conversation_ids = conv_sample['conv_id'].tolist()

# Extract emotion labels corresponding to the conversation IDs
emotions_list = thesis_df.groupby('conv_id')['emotion_label'].unique().tolist()
emotions_list = np.concatenate(emotions_list).tolist() #Flatten the list of arrays using numpy.concatenate() and Convert the flattened numpy array to a list
test_emotions_sample = emotions_list[:15]
#print(test_emotions_sample)


'''emotion_labels = ["surprised", "excited", "angry", "proud", "sad", "annoyed",
            "grateful", "lonely", "afraid", "terrified", "guilty", "impressed",
            "disgusted", "hopeful", "confident", "furious", "anxious", "anticipating",
            "joyful", "nostalgic", "disappointed", "prepared", "jealous", "content",
            "devastated", "sentimental", "embarrassed", "caring", "trusting", "ashamed",
            "apprehensive", "faithful"] '''

openai.api_key = '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'

# Define a function to generate responses as a listener based on the speaker's utterance
def generate_listener_response(speaker_utterance):
    prompt = f"The utterance of a speaker will be given. Act as the listener and reply to the speaker.\nSpeaker: {speaker_utterance}\nListener:"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    return response.choices[0].message.content

# Create a new dataframe to store the listener responses
test22_responses = pd.DataFrame(columns=['conv_id', 'Test2.2_response'])

for index, row in test_conv_sample.iterrows():
    conv_id = row['conv_id']
    # Get the first utterance of the conversation
    first_utterance = row['utterance'][0]
    # Generate listener response for the first utterance
    listener_response = generate_listener_response(first_utterance)

    #Save the conversation ID and listener response in the dataframe
    new_row = pd.DataFrame({
        'conv_id': [conv_id],
        'Test2.2_response': [listener_response]
    })
    #Append the new row to the test22_responses DataFrame
    test22_responses = pd.concat([test22_responses, new_row], ignore_index=True)

    # Print listener response and the first utterance
    #print("Listener's response:", listener_response)
    #print("Speaker's utterance:", first_utterance)

# Save the dataframe to a CSV file
test22_responses.to_csv('test22_responses.csv', index=False)

