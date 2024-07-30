from safetensors import torch

from RAG_emotions.conversaition import Conversation
from RAG_emotions.faiss_retriver_cosine import FaissRetrieverCosine


def embed_bert_cls(texts, model, tokenizer):
    t = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


# Функция для генерации ответа
def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(**data, generation_config=generation_config)[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


# Функция для выполнения запроса RAG
def rag_query(
        user_query,
        model,
        tokenizer,
        all_embeddings,
        texts,
        model_gen,
        tokenizer_gen,
        generation_config
):
    query_embedding = embed_bert_cls(user_query, model, tokenizer)
    faiss_retriever_cosine = FaissRetrieverCosine(all_embeddings)
    top_k_indices = faiss_retriever_cosine.retrieve(query_embedding, k=5)
    relevant_texts = [texts[idx] for idx in top_k_indices]
    context = "\n".join(relevant_texts)
    conversation = Conversation()
    conversation.add_user_message(f"Контекст: {context}\nВопрос: {user_query}")
    prompt = conversation.get_prompt(tokenizer_gen)
    output = generate(model_gen, tokenizer_gen, prompt, generation_config)
    conversation.add_bot_message(output)
    return output
