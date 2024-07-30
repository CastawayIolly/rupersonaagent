import click
from RAG_LLM_dialogue.query_document import qa_chain

CYAN = "\033[96m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"


@click.command()
@click.option('--question', prompt='Your Question', default="Где ты живешь?", help='Шо ты как ты?')
def main(question):
    chain = qa_chain()
    history = []
    while True:
        result = chain({"question": question})
        click.echo(f"Answer: {result['answer']}")
        history.append((question, result['answer']))
        formatted_history = f"{BOLD}Question History:{RESET}"
        click.echo(f"{formatted_history}")
        for q, a in history:
            formatted_question = f"{BOLD}{CYAN}{q}{RESET}"
            formatted_answer = f"{BOLD}{GREEN}{a}{RESET}"
            click.echo(f"{formatted_question} -> {formatted_answer}")
        question = click.prompt('\nYour Question', default="Где ты живешь?")


if __name__ == '__main__':
    main()
