import json
import random
from jsonllm.logits_processors import NumberStoppingCriteria, OutputNumbersTokens
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from typing import Any, Dict


class JSONLLM:
    value: Dict[str, Any] = {}

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        json_schema: Dict[str, Any],
        prompt: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.json_schema = json_schema
        self.prompt = prompt

        self.number_logit_processor = OutputNumbersTokens(self.tokenizer, self.prompt)
        self.number_stop_criteria = NumberStoppingCriteria(self.tokenizer)

    # def generate(
    #     self, text: str, max_length: int = 100, forced_bos_token_id: list | None = None
    # ) -> str:
    #     # print prompt in red
    #     print("generate")
    #     print("\033[91m {}\033[00m".format(text))
    #     input_ids = self.tokenizer.encode(text, return_tensors="pt")
    #     output = self.model.generate(
    #         input_ids,
    #         max_length=max_length,
    #         num_return_sequences=1,
    #         forced_bos_token_id=forced_bos_token_id,
    #     )
    #     decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
    #     return decoded_output.strip()

    def generate_number(self, suffix="") -> float:
        print("generate_number", suffix)
        prompt = self.get_prompt() + suffix

        # print prompt in red
        #       number = model.generate(
        #     tokenizer.encode(prompt, return_tensors="pt"),
        #     max_new_tokens=5,
        #     num_return_sequences=1,
        #     logits_processor=[a],
        #     stopping_criteria=[NumberStoppingCriteria(tokenizer)],
        # )

        print("\033[91m {}\033[00m".format(prompt))
        response = self.model.generate(
            self.tokenizer.encode(prompt, return_tensors="pt"),
            max_new_tokens=5,
            num_return_sequences=1,
            logits_processor=[self.number_logit_processor],
            stopping_criteria=[self.number_stop_criteria],
        )

        response = self.tokenizer.decode(response[0], skip_special_tokens=True)

        print("\033[94m {}\033[00m".format(response))
        try:
            return float(response)
        except ValueError:
            print("ValueError")
            return

    def generate_boolean(self, suffix="") -> bool:
        prompt = self.get_prompt()
        true_token_id = self.tokenizer.encode("true", add_special_tokens=False)[0]
        false_token_id = self.tokenizer.encode("false", add_special_tokens=False)[0]

        response = self.generate(
            prompt, forced_bos_token_id=[true_token_id, false_token_id]
        ).lower()

        if response == "true":
            return True
        else:
            return False

    def generate_array(
        self, item_schema: Dict[str, Any], obj: Dict[str, Any], suffix=""
    ) -> list:
        array_length = random.randint(0, 5)
        return [self.generate_value(item_schema, obj) for _ in range(array_length)]

    # add stopping criteria with "
    def generate_string(self) -> str:
        prompt = self.get_prompt()
        response = self.generate(prompt)
        return response

    def generate_object(
        self, properties: Dict[str, Any], obj: Dict[str, Any], suffix=""
    ) -> Dict[str, Any]:
        print("generate_object", properties)

        for key, schema in properties.items():
            value = self.generate_value(schema, obj, suffix=f'"{key}": ')

            obj[key] = value
        return obj

    def generate_value(self, schema: Dict[str, Any], obj: Dict[str, Any], suffix=""):
        schema_type = schema["type"]
        if schema_type == "number":
            return self.generate_number(suffix=suffix)
        elif schema_type == "boolean":
            return self.generate_boolean(suffix=suffix)
        elif schema_type == "array":
            return self.generate_array(schema["items"], obj, suffix=suffix)
        elif schema_type == "object":
            return self.generate_object(schema["properties"], obj, suffix=suffix)
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def get_prompt(self):
        template = """{prompt}\nMake sure to output in the following format:\n{schema}\n {progress}"""
        progress = json.dumps(self.value)

        progress = progress.rstrip("}").rstrip("]").rstrip(",")

        prompt = template.format(
            prompt=self.prompt,
            schema=json.dumps(self.json_schema),
            progress=progress,
        )

        return prompt

    def __call__(self) -> Dict[str, Any]:
        self.value = {}
        generated_data = self.generate_object(
            self.json_schema["properties"], self.value
        )
        return generated_data


model_name = "EleutherAI/gpt-neo-1.3B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

weather_schema = {
    "type": "object",
    "properties": {
        "temperature": {"type": "number"},
        # "humidity": {"type": "number"},
    },
}


jsonllm = JSONLLM(
    model=model,
    tokenizer=tokenizer,
    json_schema=weather_schema,
    prompt="Generate a weather object",
)

output = jsonllm()
print(output)
