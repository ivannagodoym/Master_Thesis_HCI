
import pandas as pd
import numpy as np
import openai

# Set pandas display options to show all rows and columns
pd.set_option('display.max_colwidth', None)

#Data Processing:
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
thesis_df = pd.read_csv(f"{data_path}/test2_data_sample500.csv", sep=",",on_bad_lines='skip')
test = thesis_df.head(10)

openai.api_key = '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'

# Define a function to generate responses as a listener based on the speaker's utterance
def generate_listener_response(speaker_utterance):
    prompt = f"The utterance of a speaker will be given. Act as the listener and reply to the speaker.\nSpeaker: {speaker_utterance}\nListener:"
    response = openai.chat.completions.create(
        model="gpt-4-turbo-2024-04-09",
        messages=[
            {"role": "system", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].message.content

# Create a new dataframe to store the listener responses
test22_responses = pd.DataFrame(columns=['conv_id', 'Test2.2_response_GPT4'])

for index, row in thesis_df.iterrows(): #test_conv_sample.iterrows():
    conv_id = row['conv_id']
    first_utterance = row['speaker_utterance']
    listener_response = generate_listener_response(first_utterance)
    print("Listener response:", listener_response)

    new_row = pd.DataFrame({
        'conv_id': [conv_id],
        'Test2.2_response_GPT4': [listener_response]
    })
    test22_responses = pd.concat([test22_responses, new_row], ignore_index=True)

# Save the dataframe to a CSV file
test22_responses.to_csv('test22_responses_GPT4.csv', index=False)

