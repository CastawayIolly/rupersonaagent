from threading import Lock
from typing import Optional, Tuple

import gradio as gr
from RAG_LLM_dialogue.query_document import qa_chain


class ChatWrapper:
    def __init__(self):
        self.lock = Lock()

    def __call__(self, inp: str, history: Optional[Tuple[str, str]]):
        self.lock.acquire()
        try:
            history = history or []
            multiquery_ss_results = relevant_doc.get_data(inp)
            print(multiquery_ss_results)
            ranked_documents = relevant_doc.get_ranked_documents(inp, multiquery_ss_results)
            print(ranked_documents)
            context = relevant_doc.get_context(ranked_documents)
            chain = qa_chain()
            persona = ['Я пенсионерка.', 'Я хорошо готовлю.', 'У меня двое внуков.', 'Я работала инженером.']
            output = chain.invoke({"question": inp, "persona": persona, "context": context})
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history


chat = ChatWrapper()
theme = gr.themes.Default(primary_hue="slate", secondary_hue="violet").set(body_background_fill="#ede8f7")
css = """
.submit { background-color: white !important; font-size: 20px !important; border: 1px solid #a075fc !important; }
.textbox { font-size: 20px !important; border: 1px solid #a075fc !important; }
.chatbot_row { border: 1px solid #a075fc !important; }
"""
with gr.Blocks(theme=theme, css=css, elem_classes="block") as demo:
    with gr.Row():
        gr.Markdown("<h3><center>Персонифицированный диалоговый агент</center></h3>")
    with gr.Row(elem_classes="chatbot_row"):
        chatbot = gr.Chatbot(elem_classes="Диалоговый агент")
    with gr.Row():
        message = gr.Textbox(
            label="Спрашивайте",
            placeholder="Напишите сообщение.....",
            lines=1,
            scale=5,
            elem_classes="textbox"
        )
        submit = gr.Button(value="Send", size="sm", scale=1, elem_classes="submit")
    gr.Examples(
        examples=[
            "Кто ты из феечек Winx?",
            "Расскажи немного о себе",
            "Что делаешь по вечерам?"],
        inputs=message,
    )
    state = gr.State()
    agent_state = gr.State()
    submit.click(chat, inputs=[message, state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state], outputs=[chatbot, state])
demo.launch(debug=True)
