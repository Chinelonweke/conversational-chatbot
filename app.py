import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up Streamlit UI
st.set_page_config(page_title="Conversational medical Chatbot")
st.header("Hey, Let's Chat")

# Initialize the ChatGroq model
chat = ChatGroq(temperature=0.5)

# Initialize session state for messages
if "flowmessages" not in st.session_state:
    st.session_state["flowmessages"] = [
        SystemMessage(content="medical AI chatbot")
    ]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to process responses from ChatGroq
def get_chatmodel_response(question):
    st.session_state["flowmessages"].append(HumanMessage(content=question))
    answer = chat(st.session_state["flowmessages"])
    st.session_state["flowmessages"].append(AIMessage(content=answer.content))
    return answer.content

# Function to prepare chat history
def prepare_chat_history(messages):
    chat_history = []
    for message, kind in messages:
        if kind == "ai":
            message = AIMessage(message)
        elif kind == "user":
            message = HumanMessage(message)
        chat_history.append(message)
    return chat_history

# Display existing chat messages
for message, kind in st.session_state.messages:
    with st.chat_message(kind):
        st.markdown(message)

# Chat input field
prompt = st.chat_input("Ask your questions ...")

if prompt:
    # Add user message to session state and display
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append([prompt, "user"])

    # Generate AI response
    with st.spinner("Generating response..."):
        chat_history = prepare_chat_history(st.session_state.messages)
        response = get_chatmodel_response(prompt)

    # Display AI response
    st.chat_message("ai").markdown(response)
    st.session_state.messages.append([response, "ai"])
