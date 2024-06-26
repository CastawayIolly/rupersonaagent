import getpass

import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

df = pd.read_csv('C:/Users/alinz/OneDrive/Рабочий стол/RAG_with_LLm/data/facts.csv',index_col=False)

loader = DataFrameLoader(df, page_content_column='characteristic_3')
raw_documents = loader.load()

print("Document loaded using DataFrameLoader from LangChain")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, 
    chunk_overlap=100
    )

documents = text_splitter.split_documents(raw_documents)
print(documents)
print("Text splitted using text_splitter from LangChain")

inference_api_key = getpass.getpass("Enter your HF Inference API Key:\n\n")

model_name = 'cointegrated/rubert-tiny2'
embeddings = HuggingFaceEmbeddings(model_name=model_name)

vectorstore  = FAISS.from_documents(documents, embeddings)
print("created a vectorestore using FAISS from LangChain")

vectorstore.save_local("faiss_index")
print("Vectorestore saved in 'faiss_index'")
