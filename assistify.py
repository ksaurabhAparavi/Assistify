import os
from pymilvus import MilvusClient
from pymilvus import utility
from langchain_core.prompts import ChatPromptTemplate
import json
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import (
    SystemMessage,
    HumanMessage,
)
from pymilvus import connections, utility, MilvusException, MilvusClient, DataType, Collection
import streamlit as st

# Authentication not enabled
client = MilvusClient("http://localhost:19530")
# Authentication enabled with the root user
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
    db_name="default"
)
for c in client.list_collections():
    print(client.describe_collection(collection_name=c))
    print(client.get_collection_stats(collection_name=c))

# make a connection to the milvus localhost
uri=f"http://localhost:19530"

# Establish a connection // TODO: Revise alternative setup 
connections.connect(
    uri=uri 
)  

# Init the store
client = MilvusClient(
    uri = uri,
    timeout= 60
)


client.list_collections() 
client.describe_collection('luba_qa')

# make the necessary imports
from sentence_transformers import SentenceTransformer

# initialize the model 
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

collection    = 'luba_qa'
output_fields = ['objectId','nodeId','parent','permissionId','isDeleted','chunk','content']
limit         = 10 

os.environ["OPENAI_API_KEY"] = "OPENAI_KEY"

# initialize OpenAi to return a good answer with the augmented prompt
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

# augment the user prompt with the knowledge retrieved from the vector DB
def augment_prompt(query: str, context):

    # feed into an augmented prompt
    augmented_prompt = f""" 

    Using the context below pleasse answer this query. 
        
    Contexts:
    {context}

    Query: {query}"""
    return augmented_prompt


collection    = 'luba_qa'
# Function to handle chatbot response
def chatbot_response(query):
    # create the embedding vecor of the query
    queryVectors = [model.encode(query, show_progress_bar = False)]

    # Perform the search
    points = client.search(
        collection_name=collection,
        data = queryVectors,
        limit=10,
        output_fields = ['objectId','nodeId','parent','permissionId','isDeleted','chunk','content']						
    )
    messages = [
        SystemMessage(content="You are a helpful assistant. Focus on the context provided."),
        HumanMessage(content=augment_prompt(query, points))
    ]
    response = chat(messages)
    print("STREWE:\n----------------------\n", response.content)

    return response.content

# Streamlit app layout
st.title("Assistify: Support customer chatBot")

# Session state to store chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant","content":"how can I help?"}];

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:= st.chat_input():
    st.session_state.messages.append({"role":"user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = chatbot_response(prompt)
    msg = response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)


