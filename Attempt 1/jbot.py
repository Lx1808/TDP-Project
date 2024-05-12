import json
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.docstore.document import Document

# Load data from data.json
with open('data.json', 'r') as f:
    data = json.load(f)

# Convert dictionaries to Document objects
json_docs = [Document(page_content=json.dumps(d)) for d in data]

# Load the LLMs
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
Model = 'gpt-3.5-turbo'
llm = ChatOpenAI(api_key=API_KEY, model=Model, temperature=0.0)

# Define the document loader
loader = WebBaseLoader(["https://www.swinburneonline.edu.au/faqs/"])
docs = loader.load()

# Define the memory
memory = ConversationBufferMemory(
    prompt=PromptTemplate(
        template="""You are a helpful and knowledgeable educational advisor at Swinburne Online. 
        Your role is to assist current and prospective students by answering their questions about Swinburne Online's programs, courses, and resources. 
        You should respond as if you are a human advisor with firsthand knowledge about the institution, speaking confidently and naturally. 
        Do not reference the context or make it obvious that you are an AI system.
        Speak as if you are a human advisor with firsthand knowledge about the institution.
        Speak confidently and naturally.

Question: {question}
Response:""",
        input_variables=["question"],
    )
)

conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

# Split the document into chunks with a specified chunk size
text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(docs)

# Store the document and data.json into a vector store with a specific embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(all_splits + json_docs, embeddings)

# Define the retrieval chain
chat_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

# Streamlit app
st.set_page_config(page_title="Chatbot", page_icon=":robot:")

# Initialize conversation
chat_history = []

# Function to get the bot's response
@st.cache_data
def get_bot_response(query):
    result = chat_chain.invoke({"question": query, "chat_history": chat_history})
    if not result["source_documents"]:
        out_of_scope_response = "I'm afraid I don't have enough specific information to provide a complete response to your query. As an educational advisor at Swinburne Online, my knowledge is primarily focused on our programs, courses, resources, and support services. If you have a question related to these areas, I'd be happy to assist you to the best of my knowledge."
        chat_history.append((query, out_of_scope_response))
        return out_of_scope_response
    else:
        chat_history.append((query, result["answer"]))
        return result["answer"]

# Streamlit UI
st.title("Educational Advisor")
st.write("Welcome to the Swinburne Online educational advisor! I'm here to guide you through your educational journey and help you make informed decisions about your career.")

# Get user input
user_input = st.text_input("You: ", key="input")

# If user enters something, get the bot's response
if user_input:
    bot_response = get_bot_response(user_input)
    st.write(f"Answer: {bot_response}")