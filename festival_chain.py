import json
import langchain
import transformers
import torch
from transformers import pipeline
import streamlit as st
# Load festival data
with open('festival_data.json', 'r') as f:
    festival_data = json.load(f)

# Set up Hugging Face model
nlp_model = pipeline('text-generation', model='gpt2')

# Define FestivalChain
class FestivalChain:
    def __init__(self, festival_data, nlp_model):
        self.festival_data = festival_data
        self.nlp_model = nlp_model

    def run(self, festival, topic):
        # Check if festival and topic exist
        if festival in self.festival_data and topic in self.festival_data[festival]:
            info = self.festival_data[festival][topic]
        else:
            info = f"Sorry, I don't have information about {topic} of {festival}."

        # Generate response using Hugging Face model
        prompt = f"Here is some information about the {topic} of {festival}: {info}"
        response = self.nlp_model(prompt, max_length=1000, num_return_sequences =1)
        return response[0]['generated_text']

# Create FestivalChain instance
festival_chain = FestivalChain(festival_data, nlp_model)

# Streamlit app
st.title("Festival Chatbot")

# Input fields
festival = st.text_input("Enter a festival name:")
topic = st.text_input("Enter a topic (e.g., history, significance):")

if festival and topic:
    # Get the response
    response = festival_chain.run(festival, topic)
    st.write(response)
else:
    st.write("Please enter both a festival name and a topic.")






