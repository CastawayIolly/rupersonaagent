from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch


class FiDT5(T5ForConditionalGeneration):
    """
    Modification of T5 model that can use Fusion-in-Decoder method
    """
    def __init__(self, config):
        super().__init__(config)

        self.do_fid = False

    def make_fid_encoder(self):
        self.encoder = FiDEncoder(self.encoder)

    def make_base_encoder(self):
        self.encoder = self.encoder.encoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        labels=None,
        use_cache=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # If we use Fusion-in-Decoder method
            # Change shape to (B * N) * L
            # B - Batch size
            # N - Number of passages
            # L - Length of passages
            if self.do_fid:
                input_ids = input_ids.view(self.bsz * self.n_passages, -1)
                if attention_mask is not None:
                    attention_mask = attention_mask.view(self.bsz * self.n_passages, -1)

            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Change shape back to B * (N * L) if used FiD method
            if self.do_fid:
                encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.view(self.bsz, self.n_passages * self.passage_length, -1)
                if attention_mask is not None:
                    attention_mask = attention_mask.view(self.bsz, self.n_passages * self.passage_length)

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=hidden_states,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )


class FiDEncoder(torch.nn.Module):
    """
    T5 Encoder that uses Fusion-in-Decoder method
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.do_fid = False

        self.config = self.encoder.config
        self.main_input_name = self.encoder.main_input_name

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        **kwargs
    ):
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # If we use Fusion-in-Decoder method
            # Change shape to (B * N) * L
            # B - Batch size
            # N - Number of passages
            # L - Length of passages
            if self.do_fid:
                input_ids = input_ids.view(self.bsz * self.n_passages, -1)
                if attention_mask is not None:
                    attention_mask = attention_mask.view(self.bsz * self.n_passages, -1)

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Change shape back to B * (N * L) if used FiD method
            if self.do_fid:
                encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.view(self.bsz, self.n_passages * self.passage_length, -1)
                if attention_mask is not None:
                    attention_mask = attention_mask.view(self.bsz, self.n_passages * self.passage_length)

        return encoder_outputs

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.encoder.embed_tokens = new_embeddings
