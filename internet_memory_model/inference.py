import os
from argparse import ArgumentParser

import in_model as im
import torch
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm


def add_cmdline_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--save_name',
        type=str,
        default='f-t5-big-internet',
        help='Path where trained LoRA adapter should be placed'
    )
    parser.add_argument(
        "--retriever_name",
        type=str,
        default='rubert-base-retriever',
        help='Name of Retriever model'
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='f-t5-big',
        help='Model name. Can be a name from Huggingface Hub'
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default='pretrained_models',
        help='Path to folder with pretrained model'
    )
    parser.add_argument(
        "--passage_max_length",
        type=int,
        default=128,
        help='Maximum length of passages in knowledge task'
    )
    parser.add_argument(
        "--labels_max_length",
        type=int,
        default=128,
        help='Maximum length of labels in tokens'
    )
    parser.add_argument(
        "--passage_count",
        type=int,
        default=5,
        help="Number of passages per 1 context sample in knowledge task"
    )

    return parser.parse_args()


def predict(
    context,
    personas,
    model,
    retriever,
    r_tokenizer,
    passage_max_length,
    passage_count
):
    """
    Make the full model's process from getting the search query
    to getting the final response
    """
    search_query = get_search_query(
        model, context, personas
    )
    print(search_query)

    if im.NO_QUERY not in search_query:
        knowledge_passages = get_data_from_external_sources(
            model, retriever, r_tokenizer,
            search_query, context,
            passage_max_length, passage_count
        )

        knowledge = extract_knowledge(
            model, knowledge_passages, context
        )
    else:
        knowledge = im.NO_KNOWLEDGE

    print(knowledge)

    return get_response(
        model, context, personas, knowledge
    )


def get_response(
    model,
    context,
    personas,
    knowledge
):
    """
    Generate the final response
    """
    tokenizer = model.tokenizer
    device = model.M.device
    response_ids = tokenizer(
        "<LM>" + context + personas + knowledge + im.RESPONSE_TASK,
        return_tensors='pt'
    ).input_ids.to(device)

    _, _, response = model.do_forward(
        "Inference response",
        input_ids=response_ids,
        do_fid=False,
        do_generation=True,
        penalty_alpha=0.6,
        top_k=4
    )

    return tokenizer.decode(response[0], skip_special_tokens=True)


def extract_knowledge(
    model,
    knowledge_passages,
    context
):
    """
    Extract knowledge
    """
    tokenizer = model.tokenizer
    device = model.M.device

    knowledge_passages[-1] += im.KNOWLEDGE_TASK

    knowledge_ids = tokenizer(
        ["<LM>" + context + i for i in knowledge_passages],
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).input_ids.to(device)

    knowledge_ids = knowledge_ids.unsqueeze(0)

    _, _, knowledge = model.do_forward(
        "Inference knowledge",
        input_ids=knowledge_ids,
        do_fid=True,
        do_generation=True
    )
    return tokenizer.decode(knowledge[0], skip_special_tokens=True)


def get_data_from_external_sources(
    model,
    retriever,
    tokenizer,
    search_query,
    context,
    passage_max_length,
    passage_count
):
    """
    Get data from the Internet and long-term memory.
    Retrieve the most relevant candidates from it.
    """
    device = retriever.device
    search_result = model.search_module.do_search(search_query)

    passage_ids = tokenizer(
        search_result,
        max_length=passage_max_length,
        padding=True,
        truncation=True,
        return_tensors='pt'
    ).input_ids.to(device)
    context_ids = tokenizer(
        "<LM>" + context,
        max_length=2 * passage_max_length,
        truncation=True,
        return_tensors='pt'
    ).input_ids.to(device)

    result = get_relevant_passages(
        passage_ids, context_ids,
        retriever, passage_count
    )

    return [
        tokenizer.decode(i, skip_special_tokens=True)
        for i in result[0][0]
    ] + result[1]


def get_search_query(
    model,
    context,
    personas
):
    """
    Generate the search query
    """
    tokenizer = model.tokenizer
    device = model.M.device

    search_ids = tokenizer(
        "<LM>" + context + personas + im.SEARCH_TASK,
        return_tensors='pt'
    ).input_ids.to(device)

    _, _, search_query = model.do_forward(
        "Inference search",
        input_ids=search_ids,
        do_generation=True
    )

    return tokenizer.decode(search_query[0], skip_special_tokens=True)


def get_relevant_passages(
    passage_ids,
    context_ids,
    retriever,
    top_N
):
    """
    Retrieve top_N most relevant texts from
    the Internet search and long-term memory
    """
    print(passage_ids.shape)
    q_pred = retriever(input_ids=context_ids).pooler_output

    # Process data from Internet
    p_pred = None
    for i in tqdm(range(0, passage_ids.shape[0])):
        batch = passage_ids[i].unsqueeze(0)
        ans = retriever(input_ids=batch).pooler_output

        if p_pred is None:
            p_pred = ans
        else:
            p_pred = torch.cat((p_pred, ans))

    similarity = torch.mm(q_pred, p_pred.transpose(0, 1))
    softmax_score = torch.nn.functional.log_softmax(similarity, dim=-1)

    score, idx = torch.topk(softmax_score, min(passage_ids.shape[0], top_N))
    print(score)

    # Process embeddings from memory if exists
    if not retriever.without_memory:
        similarity = torch.mm(q_pred, retriever.memory.transpose(0, 1))
        softmax_score = torch.nn.functional.log_softmax(similarity, dim=-1)

        score, mem_idx = torch.topk(softmax_score, min(retriever.memory.shape[0], top_N))
        print(score)

        texts = [retriever.texts[i] for i in mem_idx.detach().cpu().numpy()[0]]
    else:
        texts = []

    return passage_ids[idx], texts


def load_from_memory(retriever):
    """
    Load data from memory source
    """
    path = "retriever/data/memory/embeddings.csv"
    if not os.path.exists(path):
        retriever.without_memory = True
    else:
        retriever.without_memory = False
        data = Dataset.from_csv("retriever/data/memory/embeddings.csv")
        retriever.memory = torch.from_numpy(data.to_pandas().to_numpy()).to(torch.float)
        retriever.memory = retriever.memory.to(retriever.device)

        retriever.texts = Dataset.from_csv("retriever/data/memory/texts.csv")['0']


def main(args):
    model = im.InternetModel(
        args.model_dir,
        args.model_name,
        args.save_name,
        args.labels_max_length
    )
    model.load_lora_adapter()
    model.M.make_fid_encoder()
    model.M.eval()

    retriever = AutoModel.from_pretrained(os.path.join("retriever/models", args.retriever_name))
    r_tokenizer = AutoTokenizer.from_pretrained(os.path.join("retriever/models", args.retriever_name))
    retriever.eval()

    # retriever.to(device)

    load_from_memory(retriever)

    context = ""
    persona1 = []
    persona2 = []

    print("Your persona:")
    [print("    " + i) for i in persona1]
    print("_" * 15)
    print("Bot persona:")
    [print("    " + i) for i in persona2]
    print("_" * 15)

    personas = im.PERSONA1 + im.PERSONA1.join(persona1) + im.PERSONA2 + im.PERSONA2.join(persona2)

    user_in = input("User: ")
    while user_in != "stop":
        context += im.USER1_REPLY + user_in + '\n'

        response = predict(
            context,
            personas,
            model,
            retriever,
            r_tokenizer,
            args.passage_max_length,
            args.passage_count
        )

        print(f"        Bot: {response}")
        context += im.USER2_REPLY + response + '\n'
        user_in = input("User: ")


if __name__ == "__main__":
    args = add_cmdline_args()

    main(args)
