---
license: mit
datasets:
  - IlyaGusev/ru_turbo_alpaca
  - IlyaGusev/ru_turbo_alpaca_evol_instruct
  - IlyaGusev/ru_turbo_saiga
  - IlyaGusev/ru_sharegpt_cleaned
  - IlyaGusev/oasst1_ru_main_branch
  - IlyaGusev/gpt_roleplay_realm
  - lksy/ru_instruct_gpt4
language:
  - ru
  - en
library_name: peft
pipeline_tag: conversational
tags:
  - Saiga
  - ruGPT-3.5
  - 13B
  - chat
  - lora
  - Peft
  - adapter
---

# ruGPT-3.5 13B LoRA: Adapter-Only Version

Welcome to the adapter-only version of ruGPT-3.5 13B LoRA. This model is built upon the foundation of [ruGPT-3.5-13B](https://huggingface.co/ai-forever/ruGPT-3.5-13B).

📌 Important: This model was trained using settings identical to [GigaSaiga](https://huggingface.co/IlyaGusev/gigasaiga_lora), but incorporates additional dataset.

🔗 Training code is [here](https://github.com/EvilFreelancer/ruGPT-3.5-13B-lora).

> Note: If you prefer, you can opt to use the ruGPT-3.5 13B fp16 base model.

## Code sample

```python
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL_NAME = "evilfreelancer/ruGPT-3.5-13B-lora"
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>\n"
DEFAULT_SYSTEM_PROMPT = "Ты — ruGPT-3.5, русскоязычный автоматический ассистент на 13 миллиардов параметров. Ты разговариваешь с людьми и помогаешь им."

class Conversation:
    def __init__(
        self,
        message_template=DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        start_token_id=2,
        bot_token_id=46787
    ):
        self.message_template = message_template
        self.start_token_id = start_token_id
        self.bot_token_id = bot_token_id
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def get_start_token_id(self):
        return self.start_token_id

    def get_bot_token_id(self):
        return self.bot_token_id

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self, tokenizer):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += tokenizer.decode([self.start_token_id, self.bot_token_id])
        return final_text.strip()


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt")
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()

config = PeftConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
print(generation_config)

conversation = Conversation()
while True:
    user_message = input("User: ")

    if user_message.strip() == "/reset":
        conversation = Conversation()
        print("History reset completed!")
        continue

    if user_message.strip() == "":
        continue

    conversation.add_user_message(user_message)
    prompt = conversation.get_prompt(tokenizer)
    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        generation_config=generation_config
    )
    conversation.add_bot_message(output)
    print("ruGPT-3.5:", output)
    print()
    print("==============================")
    print()
```

## Examples

```text
User: Привет! Почему трава зелёная?
ruGPT-3.5: Трава зелёная благодаря хлорофиллу - пигменту растений, который поглощает световые волны определенной длины и преобразует их в энергию химических связей. Хлорофилл также обеспечивает растения питательными веществами из почвы через фотосинтез.
```

```text
User: Придумай стишок про нейросети.
ruGPT-3.5: Нейросеть – это чудо науки,
Она способна решать любые задачи.
И если ты хочешь стать программистом,
То лучше всего изучать нейронные сети.
```

```text
User: Высота самой высокой горы на Земле?
ruGPT-3.5: Самая высокая гора на Земле находится в Гималаях и называется Эверест (8848 метров).
```

```text
User: Сколько максимум лет может прожить галопогосская черепаха?
ruGPT-3.5: Галопогосские черепахи могут жить до 100 лет и более. Они являются одними из самых долгоживущих животных на планете.
```

## 📚 Training Datasets

The datasets utilized for training this model are consistent with those used for [Saiga-2](https://github.com/IlyaGusev/rulm).

Here's the comprehensive list:

- [ru_turbo_alpaca](https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca)
- [ru_turbo_alpaca_evol_instruct](https://huggingface.co/datasets/IlyaGusev/ru_turbo_alpaca_evol_instruct)
- [ru_turbo_saiga](https://huggingface.co/datasets/IlyaGusev/ru_turbo_saiga)
- [ru_sharegpt_cleaned](https://huggingface.co/datasets/IlyaGusev/ru_sharegpt_cleaned)
- [oasst1_ru_main_branch](https://huggingface.co/datasets/IlyaGusev/oasst1_ru_main_branch)
- [gpt_roleplay_realm](https://huggingface.co/datasets/IlyaGusev/gpt_roleplay_realm)
- [ru_instruct_gpt4](https://huggingface.co/datasets/lksy/ru_instruct_gpt4)

## 🛠 Training Procedure

The following `bitsandbytes` quantization config was used during training:

- quant_method: bitsandbytes
- load_in_8bit: True
- load_in_4bit: False
- llm_int8_threshold: 6.0
- llm_int8_skip_modules: None
- llm_int8_enable_fp32_cpu_offload: False
- llm_int8_has_fp16_weight: False
- bnb_4bit_quant_type: fp4
- bnb_4bit_use_double_quant: False
- bnb_4bit_compute_dtype: float32

## ⚙️ Framework Versions

Ensure you have the following framework versions for compatibility:

- PyTorch 2.1.0
- PEFT 0.5.0
- bitsandbytes 0.41.1
- transformers 4.34.0

## Links

- https://t.me/evilfreelancer
- https://dzen.ru/evilfreelancer
