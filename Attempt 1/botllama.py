import streamlit as st
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state # Last layer embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# Function to preprocess text
def preprocess_text(text):
    processed_text = text.lower() # Convert text to lowercase
    return processed_text

# Function to compute sentence embeddings using the provided script
def compute_sentence_embeddings(sentences):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings

# Function to check if user question is out-of-scope
def is_out_of_scope(user_question, question_embeddings, questions):
    user_question_embedding = compute_sentence_embeddings([user_question])
    similarity_scores = F.cosine_similarity(user_question_embedding, question_embeddings)
    max_score = torch.amax(similarity_scores)
    return max_score < 0.8

# Function to generate an out-of-scope answer
def generate_out_of_scope_answer(user_question):
    ollama_llm = Ollama(model='llama3')
    parser = StrOutputParser()
    ollama_chain = ollama_llm | parser
    out_of_scope_prompt = f"I'm sorry, I couldn't find a relevant answer to your question: '{user_question}'. Unfortunately, my knowledge is limited, and this query is out of my scope. Please try rephrasing your question or asking something else."
    out_of_scope_answer = ollama_chain.invoke(out_of_scope_prompt)
    return out_of_scope_answer

# Function to find the most relevant question-answer pair
def find_most_relevant_question(user_question, question_embeddings, questions):
    user_question_embedding = compute_sentence_embeddings([user_question])
    similarity_scores = F.cosine_similarity(user_question_embedding, question_embeddings)
    max_index = torch.argmax(similarity_scores).item()
    most_relevant_question = questions[max_index]
    return most_relevant_question

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
    else:
        print("Warning: Skipping item with missing question or answer:", item)

# Preprocess questions
processed_questions = [preprocess_text(question) for question in questions]

# Compute sentence embeddings for questions
question_embeddings = compute_sentence_embeddings(processed_questions)

# Get user input
user_question = input("Ask your question: ")

# Preprocess user question
processed_user_question = preprocess_text(user_question)

# Check if user question is out-of-scope
if is_out_of_scope(processed_user_question, question_embeddings, questions):
    out_of_scope_answer = generate_out_of_scope_answer(user_question)
    print("Out of Scope Answer:", out_of_scope_answer)
else:
    # Find the most relevant question
    most_relevant_question = find_most_relevant_question(processed_user_question, question_embeddings, questions)

    # Get the answer for the most relevant question
    answer_index = questions.index(most_relevant_question)
    most_relevant_answer = answers[answer_index]
    print("Answer:", most_relevant_answer)