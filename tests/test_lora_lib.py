import unittest
from transformers import AutoTokenizer

from lora_lib.lora_trainer import generate_prompt, tokenize, generate_and_tokenize_prompt, train


class TestModelTraining(unittest.TestCase):

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/saiga2_7b")

    def test_generate_prompt(self):
        data_point = {
            "system": "This is the system prompt.",
            "user": "This is the user input.",
            "bot": "This is the bot response."
        }
        expected_prompt = "<s>system\nThis is the system prompt.</s><s>user\nThis is the user input.</s><s>bot\nThis is the bot response.</s>"
        self.assertEqual(generate_prompt(data_point), expected_prompt)

    def test_tokenize(self):
        prompt = "<s>system\nThis is the system prompt.</s><s>user\nThis is the user input.</s><s>bot\nThis is the bot response.</s>"
        tokenized_prompt = tokenize(prompt, self.tokenizer, CUTOFF_LEN=10, add_eos_token=False)
        self.assertIn("input_ids", tokenized_prompt)
        self.assertIn("attention_mask", tokenized_prompt)

    def test_generate_and_tokenize_prompt(self):
        data_point = {
            "system": "This is the system prompt.",
            "user": "This is the user input.",
            "bot": "This is the bot response."
        }
        tokenized_prompt = generate_and_tokenize_prompt(data_point, self.tokenizer)
        self.assertIn("input_ids", tokenized_prompt)
        self.assertIn("attention_mask", tokenized_prompt)

    def test_train_function(self):
        model_name = "IlyaGusev/saiga2_7b"
        dataset = "path_to_your_dataset"  # Replace with your dataset path
        try:
            train(model_name, dataset)
        except Exception as e:
            self.fail(f"Training function raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
