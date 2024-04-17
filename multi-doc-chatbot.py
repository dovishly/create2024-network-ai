import os
import sys
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv('.env')
DOCS_FOLDER = "docs3"  #F5 status
# DOCS_FOLDER = "docs2" #F5 config
# DOCS_FOLDER = "docs"  #tor
DATA_FOLDER = "data3" #f5 status
# DATA_FOLDER = "data2" #f5 config
# DATA_FOLDER = "data" #tor
documents = []
# Create a List of Documents from all of our files in the ./docs folder
for file in os.listdir(DOCS_FOLDER):
    if file.endswith(".pdf"):
        pdf_path = "./" + DOCS_FOLDER + "/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    elif file.endswith('.docx') or file.endswith('.doc'):
        doc_path = "./" + DOCS_FOLDER + "/" + file
        loader = Docx2txtLoader(doc_path)
        documents.extend(loader.load())
    elif file.endswith('.txt'):
        text_path = "./" + DOCS_FOLDER + "/" + file
        loader = TextLoader(text_path)
        documents.extend(loader.load())

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(separator="\n\t",chunk_size=3450, chunk_overlap=50)
documents = text_splitter.split_documents(documents)

# Convert the document chunks to embedding and save them to the vector store
vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./" + DATA_FOLDER)
vectordb.persist()

# create our Q&A chain
pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.7, model_name='gpt-4'),
    retriever=vectordb.as_retriever(search_kwargs={'k': 8}),
    return_source_documents=True,
    verbose=False
)

blue = "\033[0;36m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{blue}----------------------")
print('Watchmen Bot Activate!')
print('----------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = pdf_qa.invoke(
        {"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"])
    chat_history.append((query, result["answer"]))