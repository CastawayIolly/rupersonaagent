from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_core.output_parsers import StrOutputParser

template = """Ты персонофицированный диалоговый агент отвечай на вопрос, пользуясь следующими правилами:
у тебя есть описание {persona}, отвечай на question: {question} соответсвии с той {persona}, которая у тебя есть.
Если для ответа не хватает данных в {persona} отвечай при помощи любого факта из {context}.

Вопрос: {question}"""
QA_PROMPT = PromptTemplate(input_variables=["persona", "question", "context"], template=template)


def qa_chain():
    config = {
        'temperature': 0.8,
        'context_length': 4000,
        'max_new_tokens': 4000,
        'stream': True,
        'batch_size': 64,
        'gpu_layers': 8
    }
    llm = CTransformers(
        model='IlyaGusev/saiga_mistral_7b_gguf',
        model_file='model-q4_K.gguf',
        config=config
    )
    output_parser = StrOutputParser()
    chain = QA_PROMPT | llm | output_parser
    return chain
