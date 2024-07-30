# Класс для ведения беседы
class Conversation:
    def __init__(self, message_template="<s>{role}\n{content}</s>\n",
                 system_prompt="Ты — русскоязычный эмоциональный ассистент. Ты эмоционально разговариваешь с людьми и помогаешь им.",
                 start_token_id=2, bot_token_id=46787):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def add_bot_message(self, message):
        self.messages.append({"role": "bot", "content": message})

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()
