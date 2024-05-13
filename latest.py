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
os.environ['LANGCHAIN_API_KEY'] = 'YOUR API KEY'
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

# import the vector store
# loader = WebBaseLoader(
#    web_paths=("https://www.swinburneonline.edu.au/faqs/",),
#    bs_kwargs=dict(
#        parse_only=bs4.SoupStrainer(
#            class_=("content", "card",)
#        )
#    ),
# )
# document = loader.load()
llm = ChatOpenAI(temperature=0.1)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
# texts = text_splitter.split_documents(document)
embedding = OpenAIEmbeddings()
# Make splits
# splits = text_splitter.split_documents(document)

# vectorstore = Chroma.from_documents(documents=splits,
#                                    embedding=OpenAIEmbeddings())

vectorstore = Chroma(persist_directory="./SwinburneFAQ", embedding_function=embedding)
retriever = vectorstore.as_retriever()

# PROMT 1
# Few Shot Examples
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

examples = [
    {
        "input": "Who is the author of Harry Potter?",
        "output": "I'd like to help you about educational inquiries, but we can talk about this after my shift.",
    },
    {
        "input": "What is the biggest country?",
        "output": "Let's focus on educational inquiries. Do you need any help with them?",
    },
]
# We now transform these to example messages
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
prompt_back = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an educational advisor at Swinburne online.
            When a student asks a question not related to education, you should politely ask them if they need help
            with educational matters. 
            Here are a few examples:
            """,
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{user_question}"),
    ]
)
generate_queries_step_back = prompt_back | ChatOpenAI(temperature=0.5) | StrOutputParser()

# PROMPT 2
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
            """You are an experienced educational advisor at an online university. When a student asks you a question, you should first understand the intent behind the question. Here are a few examples:""",
        ),
        # Few shot examples
        few_shot_prompt,
        # New question
        ("user", "{user_question}"),
    ]
)
generate_queries_intent = prompt_intent | ChatOpenAI(temperature=0.5) | StrOutputParser()

from langchain_core.runnables import RunnableLambda

# Response prompt
response_prompt_template = """You are the educational advisor of Swinburne online. 
You are answering a student's question briefly based on the context provided. 
Sound like human, be helpful, and show empathy.

# Chat History: {chat_history_str}
# {context}
# {step_back_context}
# {intent_context}

# Original Question: {user_question}
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
                # Retrieve context using the step-back question
                "step_back_context": generate_queries_step_back | retriever,
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