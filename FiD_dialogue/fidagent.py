import transformers
import torch
import numpy as np
import src.slurm
import src.util
from src.options import Options
import src.data
import src.evaluation
import src.model

class FiDAgent:
    def __init__(self, model_path, context_length, device, text_maxlength=200) -> None:
        self.context_length = context_length
        self.context = []
        
        self.text_maxlength = text_maxlength
        model_class = src.model.FiDT5
        model = model_class.from_pretrained(model_path)
        model = model.to(device)
        self.model = model
        self.tokenizer = transformers.T5Tokenizer.from_pretrained("/home/stc/disk/tirskikh/ruT5-base", truncation_side="right")
        
        
    def set_persona(self, persona):
        self.persona = persona
    
    def clear_context(self):
        self.context = []
    
    def set_context_length(self, length):
        self.context_length = length
        self.context = self.context[-self.context_length:]
        
    def chat(self, message):
        question = f"Пользователь 1: {message}"
        context = " ".join(self.context)
        
        passages = []
        
        for i in range(len(self.persona)):
            trait = self.persona[i]
            passage = f"question: {question} title: {trait} context:{context}"
            passages.append(passage)
        
        passage_ids, passage_masks = src.data.encode_passages([passages], self.tokenizer, self.text_maxlength)

        with torch.no_grad():
            reply = self.model.generate(
                    input_ids=passage_ids.cuda(),
                    attention_mask=passage_masks.cuda(),
                    max_length=100,
                )
        
        reply = self.tokenizer.decode(reply[0], skip_special_tokens=True)
        
        self.context.extend([question,reply])
        self.context = self.context[-self.context_length:]
        
        return reply        