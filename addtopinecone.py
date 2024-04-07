import os
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from langchain.chains import ConversationChain, RetrievalQA
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator

# os.environ["OPENAI_API_KEY"] = "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"
load_dotenv()

loader = PyPDFLoader("data.pdf")

index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
).from_loaders([loader])

# Define out-of-scope criteria
out_of_scope_keywords = ["fashion", "food", "alcohol"]

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What's up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if user input is out of scope
    is_out_of_scope = any(keyword in prompt.lower() for keyword in out_of_scope_keywords)

    if is_out_of_scope:
        out_of_scope_response = "I can help you with career and education matters. Do you have any questions about that?"
        
        with st.chat_message("assistant"):
            st.markdown(out_of_scope_response)
            st.session_state.messages.append({"role": "assistant", "content": out_of_scope_response})
    else:
        with st.spinner("Answering..."):
            answer = index.query(prompt)

        with st.chat_message("assistant"):
            if answer:
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.markdown("Sorry, I cannot understand.")
