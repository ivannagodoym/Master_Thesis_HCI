import pandas as pd
import numpy as np
import openai

# Set pandas display options to show all rows and columns
pd.set_option('display.max_colwidth', None)

# Data Processing:
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
data_chunksize = 18  # Define the chunk size
#thesis_df_reader = pd.read_csv(f"{data_path}/thesis_data.csv", sep=",", chunksize=data_chunksize, iterator=True)

openai.api_key = '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'

# Define a function to generate responses as a listener based on the speaker's utterance
def generate_listener_response(speaker_utterance):
    prompt = f"The utterance of a speaker will be given. Act as the listener and reply to the speaker.\nSpeaker: {speaker_utterance}\nListener:"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
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

# Function to process chunks ensuring conversations are not divided
def process_chunks(num_chunks_to_process):
    responses = pd.DataFrame(columns=['conv_id', 'Test2.2_response'])
    thesis_df_reader = pd.read_csv(f"{data_path}/thesis_data.csv", sep=",", chunksize=data_chunksize)
    chunks_processed = 0
    incomplete_conversation = []

    while chunks_processed < num_chunks_to_process:
        conversation = pd.DataFrame()
        for chunk_number, chunk in enumerate(thesis_df_reader):
            if 'conv_id' not in chunk.columns:
                raise ValueError("The 'conv_id' column is required for conversation grouping.")

            if incomplete_conversation:
                chunk = pd.concat([pd.DataFrame(incomplete_conversation), chunk], ignore_index=True)
                incomplete_conversation = []

            if len(conversation) == 0:
                conversation = chunk
            else:
                conversation = pd.concat([conversation, chunk])

            unique_conversations = conversation['conv_id'].nunique()
            if unique_conversations <= 0:
                continue

            if len(conversation) >= data_chunksize or unique_conversations > 1:
                chunk_responses, incomplete_conversation = process_chunk(conversation)
                responses = pd.concat([responses, chunk_responses], ignore_index=True)
                conversation = pd.DataFrame()
                chunks_processed += 1
                if chunks_processed >= num_chunks_to_process:
                    break
        # Reset the iterator to read from the beginning of the file
        thesis_df_reader = pd.read_csv(f"{data_path}/thesis_data.csv", sep=",", chunksize=data_chunksize)

    return responses

def process_chunk(chunky):
    conv_sample = chunky.groupby('conv_id')['utterance'].apply(list).reset_index().sort_values(by='conv_id')
    # Create a new dataframe to store the listener responses
    responses_chunk = pd.DataFrame(columns=['conv_id', 'Test2.2_response'])
    incomplete_conversation = []

    for index, row in conv_sample.iterrows():
        conv_id = row['conv_id']
        conversation_context = chunky.loc[chunky['conv_id'] == conv_id, 'context'].iloc[0]
        # Extract first utterance for the listener response
        first_utterance = row['utterance'][0]
        # Generate listener response
        listener_response = generate_listener_response(first_utterance)
        # Save the conversation ID and listener response in the dataframe
        new_row = pd.DataFrame({
            'conv_id': [conv_id],
            'Test2.2_response': [listener_response]
        })
        responses_chunk = pd.concat([responses_chunk, new_row], ignore_index=True)

        if len(row['utterance']) > 1:
            incomplete_conversation = [{'conv_id': conv_id, 'utterance': row['utterance'][1:]}]

    return responses_chunk, incomplete_conversation


num_chunks_to_process = 3

# Process data in chunks
responses = process_chunks(num_chunks_to_process)

# Save the dataframe to a CSV file
responses.to_csv('test22_responses_chunked.csv', index=False)

# Process data in chunks
#responses = pd.DataFrame(columns=['conv_id', 'Test2.2_response'])

#num_chunks_to_process = 24
#total_chunks_processed = 0

'''for chunk_number, chunk in enumerate(thesis_df_reader):
    if chunk_number >= num_chunks_to_process:
        break
    total_chunks_processed += 1
    # Grouping by conv_id
    conv_sample = chunk.groupby('conv_id')['utterance'].apply(list).reset_index().sort_values(by='conv_id')
    # Create a new dataframe to store the listener responses
    chunk_responses = pd.DataFrame(columns=['conv_id', 'Test2.2_response'])

    for index, row in conv_sample.iterrows():
        conv_id = row['conv_id']
        # Get the first utterance of the conversation
        first_utterance = row['utterance'][0]
        # Generate listener response for the first utterance
        listener_response = generate_listener_response(first_utterance)

        # Save the conversation ID and listener response in the dataframe
        new_row = pd.DataFrame({
            'conv_id': [conv_id],
            'Test2.2_response': [listener_response]
        })
        chunk_responses = pd.concat([chunk_responses, new_row], ignore_index=True)

    responses = pd.concat([responses, chunk_responses], ignore_index=True)

# Save the dataframe to a CSV file
responses.to_csv('test22_responses_chunked.csv', index=False)'''



##ORIGINAL CHUNKED 2.1
'''import pandas as pd
import numpy as np
import openai

# Set pandas display options to show all rows and columns
pd.set_option('display.max_colwidth', None)

# Data Processing:
data_path = "/Users/ivannagodoymunoz/Desktop/Master Thesis/Testing"
data_chunksize = 18  # Define the chunk size
thesis_df_reader = pd.read_csv(f"{data_path}/thesis_data.csv", sep=",", chunksize=data_chunksize, iterator=True)

openai.api_key = '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'

# Define a function to generate responses as a listener based on the speaker's utterance
def generate_listener_response(speaker_utterance):
    prompt = f"The utterance of a speaker will be given. Act as the listener and reply to the speaker.\nSpeaker: {speaker_utterance}\nListener:"
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-0125",
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

# Process data in chunks
responses = pd.DataFrame(columns=['conv_id', 'Test2.2_response'])

num_chunks_to_process = 24
total_chunks_processed = 0

for chunk_number, chunk in enumerate(thesis_df_reader):
    if chunk_number >= num_chunks_to_process:
        break
    total_chunks_processed += 1
    # Grouping by conv_id
    conv_sample = chunk.groupby('conv_id')['utterance'].apply(list).reset_index().sort_values(by='conv_id')
    # Create a new dataframe to store the listener responses
    chunk_responses = pd.DataFrame(columns=['conv_id', 'Test2.2_response'])

    for index, row in conv_sample.iterrows():
        conv_id = row['conv_id']
        # Get the first utterance of the conversation
        first_utterance = row['utterance'][0]
        # Generate listener response for the first utterance
        listener_response = generate_listener_response(first_utterance)

        # Save the conversation ID and listener response in the dataframe
        new_row = pd.DataFrame({
            'conv_id': [conv_id],
            'Test2.2_response': [listener_response]
        })
        chunk_responses = pd.concat([chunk_responses, new_row], ignore_index=True)

    responses = pd.concat([responses, chunk_responses], ignore_index=True)

# Save the dataframe to a CSV file
responses.to_csv('test22_responses_chunked.csv', index=False)'''