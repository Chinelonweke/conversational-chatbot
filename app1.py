import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from loguru import logger
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0,max_tokens=1000, model_name="gpt-3.5-turbo",streaming=True)

# Set up Streamlit UI
st.set_page_config(page_title='🤖 MediBot', layout='centered', page_icon='🤖')
st.header("🤖 Medibot Chat AI")


#file uploader inn the sidebar on the left
with st.sidebar:
    uploaded_files = st.file_uploader("Please upload your files", accept_multiple_files=True, type=None)


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