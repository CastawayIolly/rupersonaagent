import os

import bitsandbytes as bnb
import pytorch_lightning as pl
import torch
import torchmetrics
from fid import FiDT5
from peft import (prepare_model_for_kbit_training, LoraConfig,
                  get_peft_model, TaskType, PeftModel)
from transformers import (GPT2Tokenizer,
                          get_cosine_schedule_with_warmup, BitsAndBytesConfig)
from utils import SearchModule, compute_metrics, log_metrics, print_trainable_parameters

PERSONA1 = '<Персона пользователя 1>'
PERSONA2 = '<Персона пользователя 2>'
SEARCH_TASK = '<Выполнить поиск>'
KNOWLEDGE_TASK = '<Извлечь знания>'
RESPONSE_TASK = '<Сгенерировать реплику>'
NO_QUERY = '<Нет запроса>'
NO_KNOWLEDGE = '<Нет знаний>'
KNOWLEDGE = '<Знания>'
USER1_REPLY = '<Реплика пользователя 1>'
USER2_REPLY = '<Реплика пользователя 2>'
KNOWLEDGE_RESULT = '<Итоговые знания>'
SEARCH_RESULT = '<Итоговый поиск>'
RESPONSE_RESULT = '<Итоговая реплика>'

# FRED-T5 probably should work fine without special tokens
ATTR_TO_SPECIAL_TOKEN = {
    'additional_special_tokens': [
        PERSONA1, PERSONA2,
        USER1_REPLY, USER2_REPLY,
        SEARCH_TASK, KNOWLEDGE_TASK, RESPONSE_TASK,
        KNOWLEDGE, NO_KNOWLEDGE, NO_QUERY,
        KNOWLEDGE_RESULT, SEARCH_RESULT, RESPONSE_RESULT
    ]
}


class InternetModel(pl.LightningModule):
    def __init__(
        self,
        model_dir,
        model_name,
        save_name,
        labels_max_length
    ):
        super().__init__()

        model_path = os.path.join(model_dir, model_name)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        if os.path.exists(model_path):
            # Load from local path
            self.M = FiDT5.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path, eos_token="</s>", truncation_side='left')
        else:
            # Load from Huggingface Hub
            self.M = FiDT5.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, eos_token="</s>", truncation_side='left')

        self.search_module = SearchModule()

        self.lora_name = os.path.join("models", save_name)

        self.metrics = torchmetrics.MetricCollection(
            {
                'BLEU-1': torchmetrics.BLEUScore(n_gram=1),
                'BLEU-2': torchmetrics.BLEUScore(n_gram=2)
            }
        )

        self.labels_max_length = labels_max_length

        self.no_search_id = self.tokenizer.convert_tokens_to_ids(NO_QUERY)
        self.search_task_id = self.tokenizer.convert_tokens_to_ids(SEARCH_TASK)
        self.knowledge_task_id = self.tokenizer.convert_tokens_to_ids(KNOWLEDGE_TASK)
        self.response_task_id = self.tokenizer.convert_tokens_to_ids(RESPONSE_TASK)
        self.search_result_id = self.tokenizer.convert_tokens_to_ids(SEARCH_RESULT)
        self.knowledge_result_id = self.tokenizer.convert_tokens_to_ids(KNOWLEDGE_RESULT)
        self.response_result_id = self.tokenizer.convert_tokens_to_ids(RESPONSE_RESULT)

    def set_training_args(self, args):
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_warmup_steps = args.num_warmup_steps
        self.num_training_steps = args.num_training_steps
        self.batch_size = args.batch_size

    def add_lora_adapter(self):
        """
        Adds LoRA Adapter to the base model
        """
        self.M.gradient_checkpointing_enable()
        self.M = prepare_model_for_kbit_training(self.M)

        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q", "v", "wi", "wo"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        self.M = get_peft_model(self.M, config)
        print_trainable_parameters(self.M)

    def load_lora_adapter(self):
        """
        Loads trained LoRA Adapter for inference
        """
        self.M = PeftModel.from_pretrained(self.M, self.lora_name)

    def configure_optimizers(self):
        optimizer = bnb.optim.PagedAdamW8bit(
            self.M.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        for i in batch:
            search_ids, search_mask, search_labels = batch[0], batch[1], batch[2]
            knowledge_ids, knowledge_mask, knowledge_labels = batch[3], batch[4], batch[5]
            response_ids, response_mask, response_labels = batch[6], batch[7], batch[8]

            # Do search task
            if search_ids is not None:
                batch_size = search_ids.shape[0]
                s_loss, _, _ = self.do_forward(
                    "Search_train",
                    search_ids,
                    search_mask,
                    search_labels,
                    use_cache=self.use_cache
                )

            # Do knowledge task
            if knowledge_ids is not None:
                batch_size = knowledge_ids.shape[0]
                k_loss, _, _ = self.do_forward(
                    "Knowledge_train",
                    knowledge_ids,
                    knowledge_mask,
                    knowledge_labels,
                    use_cache=self.use_cache,
                    do_fid=True
                )

            # Do response task
            if response_ids is not None:
                batch_size = response_ids.shape[0]
                r_loss, _, _ = self.do_forward(
                    "Response_train",
                    response_ids,
                    response_mask,
                    response_labels,
                    use_cache=self.use_cache
                )

        loss = 0.
        count = 0
        if search_ids is not None:
            loss += s_loss
            count += 1
        if knowledge_ids is not None:
            loss += k_loss
            count += 1
        if response_ids is not None:
            loss += r_loss
            count += 1

        loss /= count

        logs = []
        logs.append({
            'name': 'lr',
            'value': self.trainer.optimizers[0].param_groups[0]["lr"],
            'on_epoch': False, 'on_step': True})

        if search_ids is not None:
            logs.append({'name': 'train_s_loss', 'value': s_loss, 'on_epoch': True, 'on_step': True})
        if knowledge_ids is not None:
            logs.append({'name': 'train_k_loss', 'value': k_loss, 'on_epoch': True, 'on_step': True})
        if response_ids is not None:
            logs.append({'name': 'train_r_loss', 'value': r_loss, 'on_epoch': True, 'on_step': True})

        log_metrics(self, logs, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        search_ids, search_mask, search_labels = batch[0], batch[1], batch[2]
        knowledge_ids, knowledge_mask, knowledge_labels = batch[3], batch[4], batch[5]
        response_ids, response_mask, response_labels = batch[6], batch[7], batch[8]

        # Do search task
        if search_ids is not None:
            s_loss, s_metrics, ans = self.do_forward(
                "Search_eval",
                search_ids,
                search_mask,
                search_labels,
                do_compute_bleu=True,
            )

            with open("out.txt", "a", encoding='utf-8') as w:
                w.write("   " + self.tokenizer.decode(ans[0]) + "\n")

        # Do knowledge task
        if knowledge_ids is not None:
            k_loss, k_metrics, ans = self.do_forward(
                "Knowledge_eval",
                knowledge_ids,
                knowledge_mask,
                knowledge_labels,
                do_compute_bleu=True,
                do_fid=True
            )

            with open("out.txt", "a", encoding='utf-8') as w:
                w.write("   " + self.tokenizer.decode(ans[0]) + "\n")

        # Do response task
        if response_ids is not None:
            r_loss, r_metrics, ans = self.do_forward(
                "Response_eval",
                response_ids,
                response_mask,
                response_labels,
                do_compute_bleu=True,
                # do_fid=True
            )

            with open("out.txt", "a", encoding='utf-8') as w:
                w.write("   " + self.tokenizer.decode(ans[0]) + "\n")

        loss = 0.
        count = 0
        if search_ids is not None:
            loss += s_loss
            count += 1
        if knowledge_ids is not None:
            loss += k_loss
            count += 1
        if response_ids is not None:
            loss += r_loss
            count += 1

        logs = []
        log_dicts = []
        if search_ids is not None:
            logs.append({'name': 'eval_s_loss', 'value': s_loss, 'on_epoch': True, 'on_step': True})
            log_dicts.append({'value': s_metrics, 'on_epoch': True, 'on_step': True})
        if knowledge_ids is not None:
            logs.append({'name': 'eval_k_loss', 'value': k_loss, 'on_epoch': True, 'on_step': True})
            log_dicts.append({'value': k_metrics, 'on_epoch': True, 'on_step': True})
        if response_ids is not None:
            logs.append({'name': 'eval_r_loss', 'value': r_loss, 'on_epoch': True, 'on_step': True})
            log_dicts.append({'value': r_metrics, 'on_epoch': True, 'on_step': True})

        log_metrics(self, logs, log_dicts)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        search_ids, search_mask, search_labels = batch[0], batch[1], batch[2]
        knowledge_ids, knowledge_mask, knowledge_labels = batch[3], batch[4], batch[5]
        response_ids, response_mask, response_labels = batch[6], batch[7], batch[8]
        # Do search task
        if search_ids is not None:
            _, s_metrics, ans = self.do_forward(
                "Search_test",
                search_ids,
                search_mask,
                search_labels,
                do_generation=True,
                do_compute_bleu=True,
                forced_token=self.search_result_id
            )

            with open("out.txt", "a", encoding='utf-8') as w:
                w.write("   " + self.tokenizer.decode(ans[0]) + "\n")

        # Do knowledge task
        if knowledge_ids is not None:
            _, k_metrics, ans = self.do_forward(
                "Knowledge_test",
                knowledge_ids,
                knowledge_mask,
                knowledge_labels,
                do_fid=True,
                do_generation=True,
                do_compute_bleu=True,
                forced_token=self.knowledge_result_id
            )

            with open("out.txt", "a", encoding='utf-8') as w:
                w.write("   " + self.tokenizer.decode(ans[0]) + "\n")

        # Do response task
        if response_ids is not None:
            _, r_metrics, ans = self.do_forward(
                "Response_test",
                response_ids,
                response_mask,
                response_labels,
                do_generation=True,
                do_compute_bleu=True,
                penalty_alpha=0.6,
                top_k=4,
                forced_token=self.response_result_id
            )

            with open("out.txt", "a", encoding='utf-8') as w:
                w.write("   " + self.tokenizer.decode(ans[0]) + "\n")

        log_dicts = []
        if search_ids is not None:
            log_dicts.append({'value': s_metrics, 'on_epoch': True, 'on_step': True})
        if knowledge_ids is not None:
            log_dicts.append({'value': k_metrics, 'on_epoch': True, 'on_step': True})
        if response_ids is not None:
            log_dicts.append({'value': r_metrics, 'on_epoch': True, 'on_step': True})

        log_metrics(self, log_dicts=log_dicts)

    def do_forward(
        self,
        step_name,
        input_ids=None,
        attention_mask=None,
        labels=None,
        use_cache=None,
        do_fid=False,
        do_compute_bleu=False,
        do_generation=False,
        penalty_alpha=None,
        top_k=None,
        forced_token=None
    ):
        M = self.M.get_base_model()
        M.do_fid = do_fid
        M.encoder.do_fid = do_fid
        if do_fid:
            M.bsz = input_ids.shape[0]
            M.n_passages = input_ids.shape[1]
            M.passage_length = input_ids.shape[2]

            M.encoder.bsz = input_ids.shape[0]
            M.encoder.n_passages = input_ids.shape[1]
            M.encoder.passage_length = input_ids.shape[2]

        if do_generation:
            if forced_token is not None:
                forced_token = [[1, forced_token]]
            answer = self.M.generate(
                input_ids=input_ids.view(input_ids.size(0), -1),
                max_new_tokens=self.labels_max_length,
                penalty_alpha=penalty_alpha,
                top_k=top_k,
                no_repeat_ngram_size=3,
                early_stopping=True,
                forced_decoder_ids=forced_token
            )

            loss = 0.
        else:
            out = self.M(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache
            )
            loss, logits = out.loss, out.logits
            if loss.isnan():
                loss = 1e-3
            answer = torch.argmax(logits, dim=-1)

        if do_compute_bleu:
            metrics, f1 = compute_metrics(self, answer, labels, generation=do_generation)
            metrics = {k + f" {step_name}": metrics[k] for k in metrics}
            metrics[f"F1 {step_name}"] = f1
            return loss, metrics, answer

        return loss, None, answer
