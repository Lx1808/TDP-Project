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
from langchain import hub
import bs4
from langchain.callbacks.manager import collect_runs
from langsmith import Client
from streamlit_feedback import streamlit_feedback

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'API KEY'
os.environ['LANGCHAIN_PROJECT'] = 'Swinburne'
os.environ['OPENAI_API_KEY'] = 'API KEY'
load_dotenv()

client = Client()


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

# Load the document
loader = WebBaseLoader(
   web_paths=("https://www.swinburneonline.edu.au/faqs/", "https://www.swinburneonline.edu.au/"),
   bs_kwargs=dict(
       parse_only=bs4.SoupStrainer(
           class_=("content", "card",)
        )
   ),
)
document = loader.load()
llm = ChatOpenAI(temperature=0.1)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_documents(document)
embedding = OpenAIEmbeddings()

vectorstore = Chroma(persist_directory="./SwinburneFAQ", embedding_function=embedding)
retriever = vectorstore.as_retriever()

# PROMPT 1
# Few Shot Examples
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

examples = [
    {
        "input": "Is there any scholarship available for international students?",
        "intent": "Concerns about affording education as an international student and seeking financial assistance opportunities.",
    },
    {
        "input": "What is the cost?",
        "intent": "Assessing the affordability and financial feasibility of attending the university.",
    },
    {
        "input": "Can I find a job after?",
        "intent": "Evaluating potential career prospects and employability after graduation.",
    },
    {
        "input": "Who is the author of Harry Potter?",
        "intent": "Out of scope: Asking about the author of a book series unrelated to Swinburne Online.",
    },
    {
        "input": "What is the capital of France?",
        "intent": "Out of scope: Inquiring about geography facts unrelated to Swinburne Online.",
    },
    {
        "input": "Tell me a joke.",
        "intent": "Out of scope: Requesting entertainment unrelated to educational matters at Swinburne Online.",
    },
    {
        "input": "Who wrote the book 'Pride and Prejudice'?",
        "intent": "Out of scope: Asking about the author of a book unrelated to Swinburne Online.",
    },
    {
        "input": "What is the largest continent in the world?",
        "intent": "Out of scope: Inquiring about geography facts unrelated to Swinburne Online.",
    },
    {
        "input": "Can you recommend a good movie to watch?",
        "intent": "Out of scope: Requesting entertainment recommendations unrelated to educational matters at Swinburne Online.",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{intent}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

prompt_intent = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an experienced educational advisor at an online university. 
            When a student asks you a question, your goal is to understand the deeper reason and intent behind their inquiry.
            Consider factors such as their concerns, goals, challenges, and aspirations related to their education and career.
            Here are a few examples of how to identify the underlying intent:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{user_question}"),
    ]
)
generate_queries_intent = prompt_intent | ChatOpenAI(temperature=0.7) | StrOutputParser()

from langchain_core.runnables import RunnableLambda

# Response prompt
response_prompt_template = """ 
If the question is out of scope or unrelated to the {context}: 
Do not provide direct answers to questions that are out of scope.
Politely and humorously explain that the topic falls outside your area of expertise as an educational advisor at Swinburne Online.
Redirect the conversation by inviting the student to ask about any career or educational matters relevant to their goals and the provided context. 

If the question is in scope:
Provide a thorough, helpful answer drawing from the relevant context.
Offer additional insights or suggestions to further assist the student with their education and career planning.
Encourage follow-up questions and express your eagerness to provide further guidance.
Use {intent_context} to provide a more tailored response based on the student's intent and the context of their inquiry.

# Intent Context: {intent_context}
# Context: {context}
# Chat History: {chat_history_str}
# Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

# get response
def get_response(query, chat_history, vectorstore):
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(query)
    context_str = "\n".join([doc.page_content for doc in docs])

    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    chain = (
            {
                # Retrieve context using the normal question
                "context": RunnableLambda(lambda x: x["user_question"]) | retriever,
                # Retrieve context using the intent statement
                "intent_context": generate_queries_intent | retriever,
                # Pass on the question
                "user_question": lambda x: x["user_question"],

                "chat_history_str": lambda x: chat_history_str,
            }
            | response_prompt
            | llm
            | StrOutputParser()
    )

    return chain.stream({
        "context": context_str,
        "intent_context": context_str,  
        "chat_history": chat_history_str,
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
            with collect_runs() as cb:
                ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history, vectorstore))
                st.session_state.run_id = cb.traced_runs[0].id
            output_file = generate_speech(ai_response)
            st.audio(output_file)

        st.session_state.chat_history.append(AIMessage(content=ai_response))

if st.session_state.get("run_id"):
    run_id = st.session_state.run_id
    feedback = streamlit_feedback(
        feedback_type="faces",
        optional_text_label="[Optional] Please provide an explanation",
        key=f"feedback_{run_id}",
    )

    score_mappings = {
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }

    if feedback:
        feedback_option = "faces"
        scores = score_mappings[feedback_option]
        score = scores.get(feedback["score"])

        if score is not None:
            feedback_type_str = f"{feedback_option} {feedback['score']}"

            feedback_record = client.create_feedback(
                run_id,
                feedback_type_str,
                score=score,
                comment=feedback.get("text"),
            )
            st.session_state.feedback = {
                "feedback_id": str(feedback_record.id),
                "score": score,
            }
        else:
            st.warning("Invalid feedback score.")
