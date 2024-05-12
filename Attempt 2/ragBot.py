import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
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
    return max_score < 0.7

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

# Preprocess questions
processed_questions = [preprocess_text(question) for question in questions]

# Compute sentence embeddings for questions
question_embeddings = compute_sentence_embeddings(processed_questions)

# Load the LLMs
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
Model = 'gpt-3.5-turbo'
gpt_llm = ChatOpenAI(api_key=API_KEY, model=Model)

# Only get content, not AI message
parser = StrOutputParser()
gpt_chain = gpt_llm | parser

# Creating the prompt template
template = """ You are Swinburne Online, an educational advisor. You are answering current and prospective student's questions about Swinburne Online. 
You do not make up any information that is not given in the context. You are polite and helpful. You are knowledgeable about Swinburne Online.
Context: {context}
Question: {question}"""
prompt = PromptTemplate.from_template(template=template)

# Streamlit app
st.title("AI-Powered Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask your question")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Preprocess user question
    processed_user_question = preprocess_text(user_question)

    # Check if user question is out-of-scope
    if is_out_of_scope(processed_user_question, question_embeddings, questions):
        # Load the contents of the JSON file
        loader = TextLoader('data.json', encoding='utf-8')
        document = loader.load()

        # Splitting into smaller chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.split_documents(document)

        # Embedding the chunks
        vector_storage = FAISS.from_documents(chunks, OpenAIEmbeddings())
        retriever = vector_storage.as_retriever()

        # Creating the chain
        result = RunnableParallel(context=retriever, question=RunnablePassthrough())
        chain = result | prompt | gpt_llm | parser

        # Generate the out-of-scope answer
        out_of_scope_answer = chain.invoke(user_question)
        with st.chat_message("assistant"):
            st.markdown(out_of_scope_answer)
        st.session_state.messages.append({"role": "assistant", "content": out_of_scope_answer})
    else:
        # Find the most relevant question
        most_relevant_question = find_most_relevant_question(processed_user_question, question_embeddings, questions)

        # Get the answer for the most relevant question
        answer_index = questions.index(most_relevant_question)
        most_relevant_answer = answers[answer_index]
        with st.chat_message("assistant"):
            st.markdown(most_relevant_answer)
        st.session_state.messages.append({"role": "assistant", "content": most_relevant_answer})
