import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
import openai
from langchain.memory import ConversationBufferMemory

# Load the OpenAI API key
load_dotenv("env2/.env")
key = os.getenv('coco_api')
openai.api_key = key

# Set up Streamlit title and prompt input
st.title("Welcome to GPT Generator")
prompt = st.text_input("Type your question or topic here:")

# Define the prompt template for generating content
content_template = PromptTemplate(
    input_variables=['prompt'],
    template='Generate content based on the following prompt: {prompt}'
)

# Set up memory for storing conversation history
content_memory = ConversationBufferMemory(input_key='prompt', memory_key='chat_history')

# Initialize the OpenAI language model
llm = OpenAI(temperature=0.9, openai_api_key=key)

# Set up the content generation chain
content_chain = LLMChain(llm=llm, prompt=content_template, verbose=True, output_key='content', memory=content_memory)

# Initialize Wikipedia API wrapper
wiki = WikipediaAPIWrapper()

# Generate content based on user prompt
if prompt:
    generated_content = content_chain.run(prompt)
    wikipedia_research = wiki.run(prompt)
    
    st.write("Generated Content:")
    st.write(generated_content)

    with st.expander('Content History'):
        st.info(content_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wikipedia_research)
