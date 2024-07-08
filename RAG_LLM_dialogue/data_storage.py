import getpass

import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_vectorstore(
    csv_path: str,
    model_name: str,
    output_dir: str,
    inference_api_key: str = None
):

    df = pd.read_csv(csv_path, index_col=False)
    loader = DataFrameLoader(df, page_content_column='value')
    raw_documents = loader.load()
    print("Document loaded using DataFrameLoader from LangChain")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )
    documents = text_splitter.split_documents(raw_documents)
    print("Text splitted using text_splitter from LangChain")

    if not inference_api_key:
        inference_api_key = getpass.getpass("Enter your HF Inference API Key:\n\n")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vectorstore = FAISS.from_documents(documents, embeddings)
    print("Created a vectorstore using FAISS from LangChain")

    vectorstore.save_local(output_dir)
    print(f"Vectorstore saved in '{output_dir}'")


if __name__ == "__main__":
    create_vectorstore(
        csv_path='RAG_LLM_dialogue/data/facts.csv',
        model_name='cointegrated/rubert-tiny2',
        output_dir='RAG_LLM_dialogue/faiss_index'
    )
