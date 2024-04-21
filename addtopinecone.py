import os
import edge_tts
import asyncio
import openai
import streamlit as st
from streamlit_chat import message
from tempfile import NamedTemporaryFile
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
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/lixiang/Documents/COS60011/natural-name-420312-86275a8a1de5.json"
load_dotenv()

loader = PyPDFLoader("/Users/lixiang/Documents/COS60011/es-docs-qa/TestFile.pdf")

index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
).from_loaders([loader])


def generate_speech(text, voice='en-US-JessaNeural', rate=20, pitch=0, output_file='output.mp3'):
    rate_str = f"{rate:+}%"
    pitch_str = f"{pitch:+}Hz"
    communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch=pitch_str)
    asyncio.run(communicate.save(output_file))
    return output_file


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

    with st.spinner("Answering..."):
        answer = index.query(prompt)
        output_file = generate_speech(answer)

    with st.chat_message("assistant"):
        if answer:
            st.markdown(answer)
            st.audio(output_file)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.markdown("Sorry, I can not understand.")
