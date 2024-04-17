import pandas as pd
import numpy as np
import openai

# Set pandas display options to show all rows and columns
pd.set_option('display.max_colwidth', None)

#Data Processing:
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
thesis_df = pd.read_csv(f"{data_path}/test2_data_sample500.csv", sep=",",on_bad_lines='skip')

openai.api_key = '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'

emotion_labels = ["surprised", "excited", "angry", "proud", "sad", "annoyed",
            "grateful", "lonely", "afraid", "terrified", "guilty", "impressed",
            "disgusted", "hopeful", "confident", "furious", "anxious", "anticipating",
            "joyful", "nostalgic", "disappointed", "prepared", "jealous", "content",
            "devastated", "sentimental", "embarrassed", "caring", "trusting", "ashamed",
            "apprehensive", "faithful"]


#Function to generate empathetic responses based on situation, emotion, and the first utterance
def generate_empathetic_response(context, emotion, first_utterance):
    prompt = f"Consider a situation and the emotion of a speaker. Situation: {context}\nEmotion: {emotion}\nAct as the listener and reply to the speaker.\nSpeaker: {first_utterance}\nListener:"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": prompt}],
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    #print(prompt)
    return response.choices[0].message.content

# Create a new dataframe to store the listener responses
test21_responses = pd.DataFrame(columns=['conv_id', 'Test2.1_response_GPT3.5'])

for index, row in thesis_df.iterrows():
    conv_id = row['conv_id']
    conversation_context = row['context']
    emotion = row['emotion_label']
    first_utterance = row['speaker_utterance']

    # Generate empathetic response
    empathetic_response = generate_empathetic_response(conversation_context, emotion, first_utterance)
    print("Empathetic response:", empathetic_response)

    # Save the conversation ID and listener response in the dataframe
    new_row = pd.DataFrame({
        'conv_id': [conv_id],
        'Test2.1_response_GPT35': [empathetic_response]
    })
    test21_responses = pd.concat([test21_responses, new_row], ignore_index=True)

# Save the dataframe to a CSV file
test21_responses.to_csv("test21_responses_GPT35.csv", index=False)
