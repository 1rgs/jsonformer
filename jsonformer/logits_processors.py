from typing import List
from transformers import PreTrainedTokenizer, LogitsWarper, StoppingCriteria
import torch

class StringStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        _,
    ) -> bool:
        if len(input_ids[0]) <= self.prompt_length:
            return False

        last_token_id = input_ids[0][-1]
        last_token = self.tokenizer.decode(last_token_id, skip_special_tokens=True)

        result = '"' in last_token

        return result


class NumberStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_length: int,
        precision: int = 3,
    ):
        self.tokenizer = tokenizer
        self.precision = precision
        self.prompt_length = prompt_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> bool:
        decoded = self.tokenizer.decode(
            input_ids[0][self.prompt_length :], skip_special_tokens=True
        )

        if decoded.count(".") > 1:
            return True

        if (
            decoded.count(".") == 1
            and len(decoded.strip().split(".")[1]) > self.precision
        ):
            return True

        if (
            len(decoded) > 1
            and any(c.isdigit() for c in decoded)
            and decoded[-1] in [" ", "\n"]
        ):
            return True

        return False
    

class EnumStoppingCriteria(StoppingCriteria):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        prompt_length: int,
        enums
    ):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.enums = enums

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> bool:
        decoded = self.tokenizer.decode(
            input_ids[0][self.prompt_length :], skip_special_tokens=True
        )

        return decoded in self.enums
    

class OutputNumbersTokens(LogitsWarper):
    def __init__(self, tokenizer: PreTrainedTokenizer, prompt: str):
        self.tokenizer = tokenizer
        self.tokenized_prompt = tokenizer(prompt, return_tensors="pt")
        vocab_size = len(tokenizer)
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)

        for _, token_id in tokenizer.get_vocab().items():
            token_str = tokenizer.decode(token_id).strip()

            if token_str == "" or (
                all(c.isdigit() or c == "." for c in token_str)
                and token_str.count(".") <= 1
            ):
                self.allowed_mask[token_id] = True

    def __call__(self, _, scores):
        mask = self.allowed_mask.expand_as(scores)
        scores[~mask] = -float("inf")

        return scores


class OutputEnumTokens(LogitsWarper):
    def __init__(self, tokenizer: PreTrainedTokenizer, enums):
        self.tokenizer = tokenizer
        vocab_size = len(tokenizer)
        self.allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)
        self.tree = self.build_tree(enums)
        self.is_first_call = True
        self.vocab_size = len(tokenizer)

    def create_mask(self, allowed_tokens):
        allowed_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for _, token_id in self.tokenizer.get_vocab().items():
            if token_id in allowed_tokens:
                allowed_mask[token_id] = True
        return allowed_mask

    def build_tree(self, enums):
        tree = {}
        for enum in enums:
            encoded_enum = self.tokenizer.encode(enum, add_special_tokens=False)
            curr_obj = tree
            for code in encoded_enum:
                if code in curr_obj.keys():
                    curr_obj = curr_obj[code]
                else:
                    curr_obj[code] = {}
                    curr_obj = curr_obj[code]
        return tree

    def __call__(self, input_ids, scores):
        if not self.is_first_call:
            self.tree = self.tree[int(input_ids[0][-1])]
        else:
            self.is_first_call = False

        allowed_tokens = self.tree.keys()

        if not len(allowed_tokens):
            raise Exception("Shouldn't happen")
        
        allowed_mask = self.create_mask(allowed_tokens)
        mask = allowed_mask.expand_as(scores)
        scores[~mask] = -float("inf")
        return scores

