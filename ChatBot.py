import os
import edge_tts
import streamlit as st
import asyncio
import speech_recognition as sr
import torch
import torch.nn.functional as F
import json
from transformers import AutoTokenizer, AutoModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

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

def preprocess_text(text):
    processed_text = text.lower() # Convert text to lowercase
    return processed_text

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state # Last layer embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def compute_sentence_embeddings(sentences):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

def compute_similarity_scores(user_input, question_embeddings, questions):
    user_input_embedding = compute_sentence_embeddings([user_input])
    similarity_scores = F.cosine_similarity(user_input_embedding, question_embeddings)
    return similarity_scores

# Load data from JSON file
with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract questions and answers from JSON data
questions = []
answers = []
for item in data:
    if "question" in item and "answer" in item:
        questions.append(item["question"])
        answers.append(item["answer"])

# Preprocess questions
processed_questions = [preprocess_text(question) for question in questions]

# Compute sentence embeddings for questions
question_embeddings = compute_sentence_embeddings(processed_questions)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="SwinburneFAQ Bot", page_icon="🤖")

st.title("SwinburneFAQ Bot")

embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory='./SwinburneFAQ', embedding_function=embedding)


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
        st.markdown(user_query)

    with st.spinner("Answering..."):
        with st.chat_message("AI"):
            ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history, vectorstore))
            output_file = generate_speech(ai_response)
            st.audio(output_file)

        st.session_state.chat_history.append(AIMessage(content=ai_response))
