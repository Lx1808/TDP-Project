import os
import edge_tts
import streamlit as st
import asyncio
import speech_recognition as sr
import torch
import torch.nn.functional as F
import json
import translator as ts
import mysql.connector
import yaml
from flask import Flask, request, jsonify, session
from transformers import AutoTokenizer, AutoModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from flask_cors import CORS
import random
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

def load_database_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'database.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def insert_or_update_question(question_text):
    config = load_database_config()
    connection = mysql.connector.connect(**config['development'])
    cursor = connection.cursor()
    cursor.execute("SELECT id, question_count FROM questions WHERE question_text = %s", (question_text,))
    existing_question = cursor.fetchone()
    if existing_question:
        question_id, question_count = existing_question
        question_count += 1
        cursor.execute("UPDATE questions SET question_count = %s WHERE id = %s", (question_count, question_id))
    else:
        cursor.execute("INSERT INTO questions (question_text, question_count) VALUES (%s, 1)", (question_text,))
    connection.commit()
    cursor.close()
    connection.close()

app = Flask(__name__)
CORS(app)

# Jerry, 已经加载了全部的问题和答案到data变量中
with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

@app.route('/get_random_questions')
def get_random_questions():
    questions = random.sample(data, 4)  # 从data中随机抽取4个问题
    questions_with_answers = [{'question': q['question'], 'answer': q['answer']} for q in questions]
    return jsonify(questions_with_answers)

#function for留言
# API端點，用於接收留言
@app.route('/comments', methods=['POST'])
def post_comment():
    content = request.json['content']
    # 加载配置并创建数据库连接
    config = load_database_config()
    conn = mysql.connector.connect(**config['development'])
    try:
        cursor = conn.cursor()
        # 插入留言内容到数据库
        sql = 'INSERT INTO comments (content) VALUES (%s)'
        cursor.execute(sql, (content,))
        conn.commit()
    finally:
        cursor.close()
        conn.close()
    return jsonify({'status': 'success', 'message': 'Comment added'})
#Jerry done


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

@app.route('/top_questions')
def top_questions():
    config = load_database_config()
    connection = mysql.connector.connect(**config['development'])
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT question_text FROM questions ORDER BY question_count DESC LIMIT 5")
    top_questions = cursor.fetchall()
    cursor.close()
    connection.close()
    return jsonify({'top_questions': top_questions})

@app.route('/get_similar_questions', methods=['POST'])
def get_similar_questions():
    data = request.get_json()
    query = data['query']
    similarity_scores = compute_similarity_scores(query, question_embeddings, processed_questions)
    similar_questions = []
    for score, question, original_question in sorted(zip(similarity_scores, processed_questions, questions), reverse=True)[:3]:
        similar_questions.append({
            'text': original_question,
            'score': float(score)
        })
    return jsonify({'similar_questions': similar_questions})

@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    enable_multilingual = data.get('enableMultiLingual', False)
    is_speech_input = data.get('isSpeechInput', False) 
    query = data.get('query', '').strip()
    print("query",query)
    if not query:
        return jsonify({'response': "It looks like you canceled the entry midway. If you have any additional questions or need to discuss further please feel free to let me know and I'll be happy to help!"})
    else:  
        chat_history = session.get('chat_history', [])
        chat_history.append({'sender': 'Human', 'content': query})
        similarity_scores = compute_similarity_scores(query, question_embeddings, processed_questions)
        max_similarity_score = max(similarity_scores)
        if max_similarity_score > 0.9:
            index = torch.argmax(torch.tensor(similarity_scores))  # 使用 torch.argmax 获取最大值索引
            similar_question_text = processed_questions[index]
            insert_or_update_question(similar_question_text)
        response = query_and_respond(query, chat_history, vectorstore)
        if(enable_multilingual):
            target_language=ts.detect_language(query)
            response=ts.translate(response,target_language)
        if(is_speech_input):
            speech_output = generate_speech(response)
            chat_history.append({'sender': 'AI', 'content': response, 'speech_output': speech_output})
            return jsonify({'response': response, 'is_speech_output': is_speech_input})
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
