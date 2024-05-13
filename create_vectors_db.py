print("start creating vectors db..........")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# import the vector store
embedding = OpenAIEmbeddings()

loader = PyPDFLoader("./data.pdf")
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_documents(document)

vectorstore = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory='./SwinburneFAQ')
vectorstore.persist()
vectorstore = None

print("creating vectors db successfully!")
