from typing import List, Union, Dict, Any
import json, re

class JsonformerNoTokens:
    value: Dict[str, Any] = {}

    def __init__(
        self,
        model: object,
        json_schema: Dict[str, Any],
        prompt: str,
        debug: bool = False,
        max_array_length: int = 10,
        max_number_tokens: int = 6,
        temperature: float = 0.1,
        max_string_token_length: int = 10,
    ):
        self.model = model
        self.json_schema = json_schema
        self.prompt = prompt

        self.generation_marker = "|GENERATION|"
        self.debug_on = debug
        self.max_array_length = max_array_length

        self.max_number_tokens = max_number_tokens
        self.temperature = temperature
        self.max_string_token_length = max_string_token_length

    def debug(self, *args, **kwargs):
        if self.debug_on:
            print(*args, **kwargs)

    def generate_number(self, temperature: Union[float, None] = None, iterations=0):
        prompt = self.get_prompt()
        self.debug("[generate_number] prompt", prompt)
        response = self.model.generate(
            prompt,
            max_new_tokens=self.max_number_tokens,
            stop=['",'],
            temperature=temperature or self.temperature,
        )
        self.debug("[generate_number] response", response)

        try:
            numbers = re.finditer(r"\d+(\.\d+)*", response)
            number = next(numbers).group(0)
            if "." in number:
                return float(response)
            return int(number)
        except (ValueError, StopIteration):
            self.debug(f"[generate_number] FAILED")
            if iterations > 3:
                raise ValueError("Failed to generate a valid number")

            return self.generate_number(
                temperature=self.temperature * 1.3, iterations=iterations + 1
            )

    def generate_boolean(self) -> bool:
        prompt = self.get_prompt()
        self.debug("[generate_boolean] prompt", prompt)

        next_most_likely_token = self.model.get_next_most_likely_token(prompt)

        if next_most_likely_token.lower() in ["true", "1"]:
            return True
        elif next_most_likely_token.lower() in ["false", "0"]:
            return False
        else:
            print("Failed to generate a valid boolean value")
            return None

    def generate_string(self) -> str:
        prompt = self.get_prompt()
        self.debug("[generate_string] prompt", prompt)
        response = self.model.generate(
            prompt,
            max_new_tokens=self.max_string_token_length,
            stop=['",'],
            temperature=self.temperature,
        )

        self.debug("[generate_string] response", response)
        split = response.split('"')
        assert len(split) >= 2

        return split[1].strip()

    def generate_object(self, properties: Dict[str, Any], obj: Dict[str, Any]) -> Dict[str, Any]:
        # self.debug("[generate_object] properties", properties)
        for key, schema in properties.items():
            obj[key] = self.generate_value(schema, obj, key)
        return obj

    def generate_value(
        self,
        schema: Dict[str, Any],
        obj: Union[Dict[str, Any], List[Any]],
        key: Union[str, None] = None,
    ) -> Any:
        schema_type = schema["type"]
        if schema_type == "number":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_number()
        elif schema_type == "boolean":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_boolean()
        elif schema_type == "string":
            if key:
                obj[key] = self.generation_marker
            else:
                obj.append(self.generation_marker)
            return self.generate_string()
        elif schema_type == "array":
            new_array = []
            obj[key] = new_array
            return self.generate_array(schema["items"], new_array)
        elif schema_type == "object":
            new_obj = {}
            if key:
                obj[key] = new_obj
            else:
                obj.append(new_obj)
            return self.generate_object(schema["properties"], new_obj)
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

    def generate_array(self, item_schema: Dict[str, Any], obj: Dict[str, Any]) -> list:
        for _ in range(self.max_array_length):
            element = self.generate_value(item_schema, obj)
            obj[-1] = element

            obj.append(self.generation_marker)
            input_prompt = self.get_prompt()
            obj.pop()
            next_most_likely_token = self.model.get_next_most_likely_token(input_prompt)
            if next_most_likely_token == "]":
                break

        return obj

    def get_prompt(self):
        template = """{prompt}\nOutput result in the following JSON schema format, one value at a time, do not continue the JSON:\n{schema}\nResult: {progress}"""
        progress = json.dumps(self.value)
        gen_marker_index = progress.find(f'"{self.generation_marker}"')
        if gen_marker_index != -1:
            progress = progress[:gen_marker_index]
        else:
            raise ValueError("Failed to find generation marker")

        prompt = template.format(
            prompt=self.prompt,
            schema=json.dumps(self.json_schema),
            progress=progress,
        )

        return prompt

    def __call__(self) -> Dict[str, Any]:
        self.value = {}
        generated_data = self.generate_object(self.json_schema["properties"], self.value)
        return generated_data
