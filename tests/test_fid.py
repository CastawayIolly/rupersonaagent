from FiD_dialogue.fidagent import FiDAgent
from FiD_dialogue.src.data import Dataset, Collator, load_data
import json
import pytest
import torch
from hamcrest import assert_that, equal_to


model_path = "/home/stc/disk/tirskikh/checkpoint/ru_experiment7/checkpoint/latest"
agent = FiDAgent(model_path=model_path,context_length=7,device="cuda:0")
persona = ["Я факт1","Я факт2","Я факт3"]
message = 'Привет'
context = ['Сообщение']*10

data_sample = { 
            'id': '0', 
            'question': 'Последняя реплика', 
            'target': 'Ожидаемый ответ', 
            'answers': ['повторяет target'], 
            'ctxs': 
                [ 
                    { 
                        "title": "Факт_о_персоне_1", 
                        "text": "История диалога" 
                    }
                ] 
            }
data_samples = [data_sample]*10

test_data_path = "/home/stc/disk/tirskikh/rupersonaagent/tests/test_data_fid.json"


@pytest.mark.gpu
class TestFidAgent:       
    def test_persona(self):
        agent.set_persona(persona)
        assert agent.persona == persona
    
    def test_context(self):
        agent.context = context
        agent.set_context_length(5)
        assert len(agent.context) == 5
        assert agent.context_length == 5
        agent.clear_context()
        assert len(agent.context) == 0
    
    def test_chat(self):
        agent.clear_context()
        response = agent.chat(message)
        assert type(response) == str
        assert len(response) > 0
        assert len(agent.context) == 2
        
        
@pytest.mark.gpu
class TestData:       
    def test_dataset(self):
        test_data = load_data(test_data_path)
        
        question_prefix='question:'
        title_prefix='title:'
        passage_prefix='context:'
        n_context=4
        
        dataset = Dataset(test_data,
                        question_prefix=question_prefix,
                        title_prefix=title_prefix,
                        passage_prefix=passage_prefix, 
                        n_context=n_context)
        
        sample = next(iter(dataset))
        
        assert len(sample) == 5
        assert len(sample['passages']) == n_context
        assert question_prefix in sample['question']
        assert title_prefix in sample['passages'][0] and passage_prefix in sample['passages'][0]
        
    def test_collator(self):
        test_data = load_data(test_data_path)
        
        dataset = Dataset(test_data, n_context=5)
        sample = next(iter(dataset))
        tokenizer = agent.tokenizer
        last_n = 5
        answer_maxlength = 20
        text_maxlength = 100
        
        collator = Collator(
                text_maxlength = text_maxlength,
                tokenizer = tokenizer,
                answer_maxlength = answer_maxlength,
                last_n=last_n
            )
        
        data = collator([sample])
        target_ids = data[1]
        passage_ids = data[3]
        
        assert len(data) == 5
        assert len(target_ids[0]) == answer_maxlength and type(target_ids) == torch.Tensor
        assert len(passage_ids[0][0]) == text_maxlength and type(passage_ids) == torch.Tensor