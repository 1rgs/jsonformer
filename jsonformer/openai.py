from typing import Optional
import os

import openai
from transformers import PreTrainedTokenizer

openai.api_key = os.getenv("OPENAI_API_KEY")


GENERATION_MARKER = "|GENERATION|"


class OpenAIModel:
    def __init__(self, model_id: str = "text-curie-001", debug: bool = False):
        self.model_id = model_id
        self.debug_on = debug

    def __request(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        logprobs: Optional[int] = None,
        **kwargs,
    ):
        response = openai.Completion.create(
            model=self.model_id,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_new_tokens,
            logprobs=logprobs,
        )
        self.debug(response)
        return response

    def __clean_initial_new_lines(self, text: str):
        # models often start responses with new lines
        while text.startswith("\n"):
            text = text[1:]
        return text

    def debug(self, *args, **kwargs):
        if self.debug_on:
            print(*args, **kwargs)

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        response = self.__request(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        return self.__clean_initial_new_lines(response.choices[0].text)

    def get_next_most_likely_token(
        self,
        prompt: str,
    ):
        response = self.__request(prompt, logprobs=1)
        for obj in response.choices[0].logprobs.top_logprobs:
            # sorted from the most likely like so: [{"token": "logprob"}]
            if "\n" in obj:
                continue
            return list(obj.keys())[0]

class OpenAITokenizer(PreTrainedTokenizer):
    def __init__(self, model_id: str, **kwargs):
        '''
        There's no word boundary tokens here.
        Therefore decoding into words is not possible, e.g. "colors" -> [2134, 4124] -> "col ors"
        '''
        import tiktoken
        from collections import OrderedDict
        super().__init__(**kwargs)
        self.tokenizer = tiktoken.encoding_for_model(model_id)
        self.__load_vocab()
        self.ids_to_tokens = OrderedDict({id: token for token, id in self.vocab.items()})

    def __load_vocab(self):
        self.vocab = {}
        for byte_token in self.tokenizer.token_byte_values():
            token_id = self.tokenizer.encode_single_token(byte_token)
            str_token = self.tokenizer.decode([token_id])
            self.vocab[str_token] = token_id

    def _tokenize(self, text: str, **kwargs):
        token_ids = self.tokenizer.encode(text)
        return [self.tokenizer.decode([token_id]) for token_id in token_ids]

    def _convert_token_to_id(self, token: str):
        return self.vocab[token]

    def _convert_id_to_token(self, id: int):
        return self.ids_to_tokens[id]

    def get_vocab(self):
        return self.vocab
