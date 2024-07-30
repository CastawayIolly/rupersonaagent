import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from RAG_LLM_dialogue.src.model import BiEncoder
from transformers import AutoTokenizer

checkpoint_path = 'bi_encoder/biencoder_checkpoint.ckpt'
model_name = 'cointegrated/rubert-tiny2'

embeddings = HuggingFaceEmbeddings(model_name=model_name)
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


def load_bi_encoder_checkpoint(checkpoint_path, model_name):
    model = BiEncoder(model_name=model_name)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_data(query):
    query_results = vectorstore.similarity_search(query)
    multiquery_ss_results = list(set([result.page_content for result in query_results]))
    return multiquery_ss_results


def get_ranked_documents(query, multiquery_ss_results):
    model, tokenizer = load_bi_encoder_checkpoint(checkpoint_path, model_name)
    with torch.no_grad():
        query_tokens = tokenizer([query] * len(multiquery_ss_results), return_tensors="pt", padding=True, truncation=True)
        query_input_ids = query_tokens['input_ids']
        query_attention_mask = query_tokens['attention_mask']
        candidate_input_ids = tokenizer(multiquery_ss_results, return_tensors="pt", padding=True, truncation=True)['input_ids']
        candidate_attention_mask = tokenizer(multiquery_ss_results, return_tensors="pt", padding=True, truncation=True)['attention_mask']
        query_embeddings, candidate_embeddings = model(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            candidate_input_ids=candidate_input_ids,
            candidate_attention_mask=candidate_attention_mask
        )
    cosine_similarities = torch.nn.functional.cosine_similarity(query_embeddings, candidate_embeddings, dim=-1)
    cosine_similarities = cosine_similarities.tolist()
    ranked_documents = sorted(zip(multiquery_ss_results, cosine_similarities), key=lambda x: x[1], reverse=True)
    return ranked_documents


def limp_rerank(ranked_documents):
    beginning_list = [ranked_documents[i] for i in range(0, len(ranked_documents), 2)]
    end_list = [ranked_documents[i] for i in range(1, len(ranked_documents), 2)][::-1]
    return beginning_list + end_list


def get_context(ranked_documents):
    query_reranked = limp_rerank(ranked_documents)
    query_reranked = [q[0] for q in query_reranked]
    context = ('\n').join(query_reranked)
    return context
