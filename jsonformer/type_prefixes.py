from transformers import PreTrainedTokenizer
from typing import Dict, List
import re

def is_number_prefix(s: str) -> bool:
    return re.match(r"^[\-\d]+\.?[\d]*$", s)

def is_boolean_prefix(s: str) -> bool:
    return 'true'.startswith(s) or 'false'.startswith(s)

def is_null_prefix(s: str) -> bool:
    return 'null'.startswith(s)

def is_string_prefix(s: str) -> bool:
    return re.match(r'^"[^"]*"?$', s)

def is_array_prefix(s: str) -> bool:
    return re.match(r'^\[["\-\d\[{]*$', s)

def is_object_prefix(s: str) -> bool:
    return re.match(r'^\{"?$', s)

def get_prefix_tokens_for_types(tokenizer: PreTrainedTokenizer) -> Dict[str, List[str]]:
    vocab = tokenizer.vocab.items()
    return {
        "number": [v for k, v in vocab if is_number_prefix(k)],
        "boolean": [v for k, v in vocab if is_boolean_prefix(k)],
        "null": [v for k, v in vocab if is_null_prefix(k)],
        "string": [v for k, v in vocab if is_string_prefix(k)],
        "array": [v for k, v in vocab if is_array_prefix(k)],
        "object": [v for k, v in vocab if is_object_prefix(k)],
    }
