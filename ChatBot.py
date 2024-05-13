import os
import edge_tts
import streamlit as st
import asyncio
import speech_recognition as sr
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import multilingual as ml
load_dotenv()


def recognize_speech():
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        audio = r.listen(source, 10, 10)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return "I can't understand you."
        except sr.RequestError:
            return "Time out!"


def generate_speech(text, voice='en-US-JessaNeural', rate=20, pitch=0, output_file='output.mp3'):
    rate_str = f"{rate:+}%"
    pitch_str = f"{pitch:+}Hz"
    communicate = edge_tts.Communicate(text, voice, rate=rate_str, pitch=pitch_str)
    asyncio.run(communicate.save(output_file))
    return output_file


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="SwinburneFAQ Bot", page_icon="🤖")

st.title("SwinburneFAQ Bot")

# import the vector store
#loader = PyPDFLoader("/Users/lixiang/Documents/GitHub/TDP-Project/data.pdf")
#document = loader.load()

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
#texts = text_splitter.split_documents(document)
embedding = OpenAIEmbeddings()

vectorstore = Chroma(persist_directory="./SwinburneFAQ", embedding_function=embedding)


# get response
def get_response(query, chat_history, vectorstore):
    template = """
        You are Swinburne Online, an educational advisor. You are answering current and prospective student's questions about Swinburne Online. You do not make up any information that is not given in the context.

        Context: {context}

        You are polite and helpful. You are knowledgeable about Swinburne Online.
        Chat history: {chat_history}
        User question: {user_question}
        """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI()

    # Retrieve relevant documents
    docs = vectorstore.similarity_search(query)
    context_str = "\n".join([doc.page_content for doc in docs])
    print("context_str:",context_str)
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "context": context_str,
        "chat_history": chat_history,
        "user_question": query
    })


# conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

speech_btn = st.button("🎤Click here to speak.")

# user input
user_query = st.chat_input("Your message")

if speech_btn:
    speech_text = recognize_speech()
    user_query = speech_text

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        print("HumanMessage(user_query)",user_query)
        language_type=ml.language_detection(user_query)
        print("language:",language_type)
        st.markdown(user_query)

    with st.spinner("Answering..."):
        with st.chat_message("AI"):
            ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history, vectorstore))
            print("ai_response:",ai_response)
            
            ai_response_translated = ml.translator(language_type,ai_response)
            print("ai_response_translated:", ai_response_translated)
            st.markdown(ai_response_translated)
            output_file = generate_speech(ai_response)
            st.audio(output_file)
        # st.session_state.chat_history.append(AIMessage(content=ai_response))
