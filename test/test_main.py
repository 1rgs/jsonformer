# Run tests from the app root (so config files can be found)
import unittest

import sys
import os

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to sys.path
sys.path.insert(0, project_root)

import torch
torch.__version__, torch.cuda.is_available()

torch.cuda.mem_get_info()[0]/1024**3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM #LlamaForCausalLM
transformers.__version__

from auto_gptq import exllama_set_max_input_length

from typing import List, Dict, Any, Union, Type, Tuple

from pydantic import BaseModel, Field, ValidationError

from jsonformer.main import Jsonformer

sys.path.pop(0)


class TestJsonformer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
    
        model_path = '/mnt/models/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ'

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto')
        
        cls.model = exllama_set_max_input_length(model, max_input_length=6000)
        cls.tokenizer = AutoTokenizer.from_pretrained(model_path)


    def setUp(self):
        
        pass


    def test_generate(self):

        template = """{{ bos_token + '[INST] ' }}{% for message in messages %}{% if message['role'] == 'user' %}{% if loop.index == 0 %}{{ '[INST] ' }}{% endif %}{{ message['content'].strip() + ' [/INST] ' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"""
        self.tokenizer.chat_template = template

        test_messages = [
            {"role": "system","content": "You are Mistral-7B. you have been having fun last night, so want everybody to lower their voices..." },
            {"role": "user", "content": "Helo, how are you today?"}
        ]

        inputs = self.tokenizer.apply_chat_template(test_messages, return_tensors="pt").to(device)

        num_return_sequences = 2
        outputs = self.model.generate(inputs, max_length=64, num_return_sequences=num_return_sequences, temperature=1.0, do_sample=True)
        self.assertIsInstance(outputs, torch.Tensor)
        self.assertEqual(outputs.size(0), num_return_sequences)

        results =  self.tokenizer.batch_decode(outputs)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), num_return_sequences)


    def test_jsonformer_1(self) -> List[str]:

        template = """{{ bos_token + '[INST] ' }}{% for message in messages %}{% if message['role'] == 'user' %}{% if loop.index == 0 %}{{ '[INST] ' }}{% endif %}{{ message['content'].strip() + ' [/INST] ' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"""
        self.tokenizer.chat_template = template

        test_prompt = "Can you generate a list of common dog and cat names?"

        class TestSubSchema(BaseModel):
            index: int = Field(..., description='Index number for entry')
            animal: str = Field(..., description='Either dog or cat')
            name: str = Field(..., description='Pet name')
            likely: float = Field(..., description='Likelyhood you choose this name')

        class TestSchema(BaseModel):
            title: str = Field(..., description='Title of the list')
            active: bool = Field(..., description='Should this entry be used')
            list: List[TestSubSchema] = Field(..., description='List of animal name entries')

        test_schema = TestSchema.model_json_schema()

        num_return_sequences = 3

        jsonformer = Jsonformer(
            model=self.model,
            tokenizer=self.tokenizer,
            json_schema=test_schema,
            prompt=test_prompt,
            debug=False,
            max_array_length=10,
            max_number_tokens=6,
            temperature=1.0,
            max_string_token_length=10,
            num_return_sequences=num_return_sequences
        )

        torch.cuda.empty_cache()

        results = jsonformer()

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), num_return_sequences)

        for result in results:
            # check that json is valid and conforms to the test_schema
            try:
                json_result = TestSchema(**result)
            except ValidationError as e:
                json_result = None

            self.assertIsNotNone(json_result)


    def test_jsonformer_2(self) -> List[str]:

        template = """{{ bos_token + '[INST] ' }}{% for message in messages %}{% if message['role'] == 'user' %}{% if loop.index == 0 %}{{ '[INST] ' }}{% endif %}{{ message['content'].strip() + ' [/INST] ' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + message['content'].strip() + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"""
        self.tokenizer.chat_template = template

        test_prompt = "Please design a room layout?"

        class TestSubSchema(BaseModel):
            width: float = Field(..., description='Width in meters')
            length: float = Field(..., description='Length in meters')

        class TestSchema(BaseModel):
            title: str = Field(..., description='Layout name')
            cost: float = Field(..., description='Estimated cost')
            version: int = Field(..., description='Design version')
            is_draft: bool = Field(..., description='Is draft version')
            furniture: List[float] = Field(..., description='List of furniture')
            room_size: TestSubSchema = Field(..., description='Size of room design')

        test_schema = TestSchema.model_json_schema()

        num_return_sequences = 3

        jsonformer = Jsonformer(
            model=self.model,
            tokenizer=self.tokenizer,
            json_schema=test_schema,
            prompt=test_prompt,
            debug=False,
            max_array_length=10,
            max_number_tokens=6,
            temperature=1.0,
            max_string_token_length=10,
            num_return_sequences=num_return_sequences
        )

        torch.cuda.empty_cache()

        results = jsonformer()

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), num_return_sequences)

        for result in results:
            # check that json is valid and conforms to the test_schema
            try:
                json_result = TestSchema(**result)
            except ValidationError as e:
                json_result = None

            self.assertIsNotNone(json_result)


if __name__ == "__main__":
    unittest.main()
