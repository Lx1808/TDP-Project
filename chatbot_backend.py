import os
import edge_tts
import streamlit as st
import asyncio
import speech_recognition as sr
import torch
import torch.nn.functional as F
import json
from flask import Flask, request, jsonify, session
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
from flask_cors import CORS

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

def recommend_similar_questions():
    user_input = st.session_state.user_input
    if not user_input:
        return []
    similarity_scores = compute_similarity_scores(user_input, question_embeddings, processed_questions)
    top_indices = similarity_scores.topk(5).indices
    top_questions = [questions[idx] for idx in top_indices]
    return top_questions



app = Flask(__name__)
CORS(app)

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


embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory='./SwinburneFAQ', embedding_function=embedding)

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    query = data['query']
    chat_history = session.get('chat_history', [])
    chat_history.append({'sender': 'Human', 'content': query})
    response = query_and_respond(query, chat_history, vectorstore)
    print('++++-----++++')
    print(response)
    chat_history.append({'sender': 'AI', 'content': response})

    return jsonify({'response': response})

def query_and_respond(query, chat_history, vectorstore):
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

    result_stream = chain.stream({
        "context": context_str,
        "chat_history": chat_history,
        "user_question": query
    })

    result_string = ''.join(result for result in result_stream)

    return result_string

if __name__ == '__main__':
    app.run(debug=True)