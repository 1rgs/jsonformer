from typing import List, Union, Dict, Any, Type

from jsonformer.logits_processors import (
    NumberStoppingCriteria,
    OutputNumbersTokens,
    StringStoppingCriteria,
)
from termcolor import cprint
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import json
import re

GENERATION_MARKER = "|GENERATION|"


class Jsonformer:

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        json_schema: Dict[str, Any],
        prompt: str,
        *,
        debug: bool = False,
        max_array_length: int = 10,
        max_number_tokens: int = 6,
        temperature: float = 1.0,
        max_string_token_length: int = 10,
        num_return_sequences: int = 1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.json_schema = json_schema
        self.prompt = prompt

        self.number_logit_processor = OutputNumbersTokens(self.tokenizer, self.prompt)

        self.generation_marker = "|GENERATION|"
        self.debug_on = debug
        self.max_array_length = max_array_length

        self.max_number_tokens = max_number_tokens
        self.temperature = temperature
        self.max_string_token_length = max_string_token_length

        ''' Support number of sequences parameter for multi-sequence generation '''
        self.num_return_sequences = num_return_sequences
        self.values = []

    def debug(self, caller: str, value: str, is_prompt: bool = False):
        if self.debug_on:
            if is_prompt:
                cprint(caller, "green", end=" ")
                cprint(value, "yellow")
            else:
                cprint(caller, "green", end=" ")
                cprint(value, "blue")


    def generate_number(self, temperature: Union[float, None] = None, precision: int = 3, num_type: Type[Union[float, int]] = float, iterations=0):

        ''' retrieve a batch of sequences of size number requested sequences '''
        prompt_batch = self.get_prompt()
        
        ''' tokenize the number of sequences as a batch at this point '''
        encoded_prompts = [self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(self.model.device) for prompt in prompt_batch]
        max_length = max(tensor.size(0) for tensor in encoded_prompts)
        input_tokens = [
            torch.nn.functional.pad(
                encoded_prompt,
                (max_length - encoded_prompt.size(0), 0),
                mode='constant',
                value=self.tokenizer.pad_token_id
            ) for encoded_prompt in encoded_prompts
        ]
        input_tokens = torch.stack(input_tokens, dim=0)
        
        precision = -1 if type is int else precision - 1 # error in stopping class always stops one too late
        
        response_batch = self.model.generate(
            inputs=input_tokens, # a batch at this point of size num_return_sequences
            max_new_tokens=self.max_number_tokens,
            num_return_sequences=1,
            logits_processor=[self.number_logit_processor],
            stopping_criteria=[
                NumberStoppingCriteria(self.tokenizer, len(input_tokens[0]), precision) # also set precision (0 for integer)
            ],
            temperature=temperature or self.temperature,
            do_sample=True, # sample from the options, otherwise temperature is in vain
            pad_token_id=self.tokenizer.eos_token_id,
        )

        ''' decode the generated tokens batch '''
        response_batch = self.tokenizer.batch_decode(response_batch, skip_special_tokens=True)
        response_batch = [response[len(prompt):] for prompt, response in zip(prompt_batch, response_batch)] # strip prompt from all generated values
        response_batch = [response.strip().rstrip(".") for response in response_batch] # strip each value in batch

        self.debug("[generate_number]", response_batch)

        values = []
        for response in response_batch:
            try:
                value = num_type(response)
            except ValueError:
                self.debug("[generate_number]", "ValueError(\"Failed to generate a valid number\")")
                value = num_type(0)
            values.append(value)

        return values


    def generate_boolean(self) -> bool:
        ''' retrieve a batch of sequences of size number requested sequences '''
        prompt_batch = self.get_prompt()
        
        self.debug("[generate_boolean]", prompt_batch, is_prompt=True)

        ''' tokenize the number of sequences as a batch at this point '''
        encoded_prompts = [self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(self.model.device) for prompt in prompt_batch]
        max_length = max(tensor.size(0) for tensor in encoded_prompts)
        input_tokens = [
            torch.nn.functional.pad(
                encoded_prompt,
                (max_length - encoded_prompt.size(0), 0),
                mode='constant',
                value=self.tokenizer.pad_token_id
            ) for encoded_prompt in encoded_prompts
        ]
        input_tokens = torch.stack(input_tokens, dim=0)

        output = self.model.forward(input_tokens)
        logits = output.logits[:, -1]  # Use all elements along the batch dimension
        
        ''' get the (first) token for true and false '''
        binary_token_ids = torch.tensor([
            self.tokenizer.encode("true", add_special_tokens=False)[0],  # Extract the first element
            self.tokenizer.encode("false", add_special_tokens=False)[0]])  # Extract the first element
        
        # Apply temperature and sample
        logits_with_temperature = logits[:, binary_token_ids] / max(self.temperature, 1e-5)
        sampled_tokens = torch.multinomial(torch.nn.functional.softmax(logits_with_temperature, dim=-1), 1)
        result = sampled_tokens[:, 0] == 0
        
        self.debug("[generate_boolean]", result)
        
        return result.cpu().numpy().tolist()
    

    def generate_string(self) -> str:
        ''' retrieve a batch of sequences of size number requested sequences '''
        prompt_batch = [prompt + '"' for prompt in self.get_prompt()]
        
        self.debug("[generate_string]", prompt_batch, is_prompt=True)

        ''' tokenize the number of sequences as a batch at this point '''
        encoded_prompts = [self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(self.model.device) for prompt in prompt_batch]
        max_length = max(tensor.size(0) for tensor in encoded_prompts)
        input_tokens = [
            torch.nn.functional.pad(
                encoded_prompt,
                (max_length - encoded_prompt.size(0), 0),
                mode='constant',
                value=self.tokenizer.pad_token_id
            ) for encoded_prompt in encoded_prompts
        ]
        input_tokens = torch.stack(input_tokens, dim=0)

        response_batch = self.model.generate(
            inputs=input_tokens, # a batch at this point of size num_return_sequences
            max_new_tokens=self.max_string_token_length,
            num_return_sequences=1,
            stopping_criteria=[
                StringStoppingCriteria(self.tokenizer, len(input_tokens[0]))
            ],
            temperature=self.temperature,
            do_sample=True, # sample from the options, otherwise temperature is in vain
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response_batch = self.tokenizer.batch_decode(response_batch[:,input_tokens.size(1):], skip_special_tokens=True)
        response_batch = self.remove_closing_quotes(response_batch)
        
        self.debug("[generate_string]", response_batch)
        
        return response_batch


    def generate_object(
        self, properties: Dict[str, Any], objects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        for key, schema in properties.items():

            ''' self.generate_value now returns a List[Any], entry per requested sequence ''' 
            values = self.generate_value(schema, objects, key)
            
            ''' set entry for each requested sequence ''' 
            for object, value in zip(objects, values):
                object[key] = value

            self.debug("[generate_object]", objects, is_prompt=True)

        return objects
    

    def substitute_schema(self, object_schema: dict) -> dict[str, Any]:
        if 'allOf' in object_schema:
            for entry in object_schema['allOf']:
                if '$ref' in entry:
                    ref_value = entry['$ref']
                    ref_parts = ref_value.split('/')
                    class_name = ref_parts[-1]

                    if '$defs' in self.json_schema:
                        class_definition = self.json_schema['$defs'].get(class_name, None)
                        if class_definition:
                            object_schema.update(class_definition)
                            self.debug('substitute_schema', f'{type(object_schema)}\n{object_schema}')

        elif '$ref' in object_schema:
            ref_value = object_schema['$ref']
            ref_parts = ref_value.split('/')
            class_name = ref_parts[-1]

            if '$defs' in self.json_schema:
                class_definition = self.json_schema['$defs'].get(class_name, None)
                if class_definition:
                    object_schema.update(class_definition)
                    self.debug('substitute_schema', f'{type(object_schema)}\n{object_schema}')

        return object_schema


    def generate_value(
        self,
        schema: Dict[str, Any],
        objects: List[Union[Dict[str, Any], List[Any]]],
        key: Union[str, None] = None,
    ) -> List[Any]:
        
        schema_type = schema.get('type',None)
        if schema_type is None and not schema.get('allOf', None) is None:
            schema_type = 'object'
        elif schema_type is None and not schema.get('$ref', None) is None:
            schema_type = 'object'

        # Common code for setting entry for each requested sequence
        def set_entry(object):
            if key:
                object[key] = self.generation_marker
            else:
                object.append(self.generation_marker)

        # Common code for setting and preparing entry for each requested sequence
        def prepare_entry(new_objects, object_type):
            result = []
            for object in objects:
                new_object = object_type()
                new_objects.append(new_object)
                set_entry(object)
                result.append(new_object)
            return result

        # process supported schema types
        if schema_type == "number":
            for object in objects:
                set_entry(object)
            return self.generate_number()

        elif schema_type == "integer":
            for object in objects:
                set_entry(object)
            return self.generate_number(precision=-1, num_type=int)

        elif schema_type == "boolean":
            for object in objects:
                set_entry(object)
            return self.generate_boolean()

        elif schema_type == "string":
            for object in objects:
                set_entry(object)
            return self.generate_string()

        elif schema_type == "array":
            new_arrays = prepare_entry([], list)
            return self.generate_array(schema["items"], new_arrays)

        elif schema_type == "object":
            schema = self.substitute_schema(schema)['properties']
            new_objects = prepare_entry([], dict)
            return self.generate_object(schema, new_objects)

        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")


    def generate_array(self, item_schema: Dict[str, Any], objects: List[Dict[str, Any]]) -> list:

        # selectors for active sequences, that didn't reach the end of the array yet
        selectors = [True for _ in objects]

        counter = 0
        for _ in range(self.max_array_length):

            if not any(selectors):
                break

            selected_objects = [object for object, select in zip(objects, selectors) if select]
            selectors = [True for select in selectors if select] # align selector list with active branches

            # forces array to have at least one element
            element_batch = self.generate_value(item_schema, selected_objects)
            
            for selected_object, element in zip(selected_objects, element_batch):
                selected_object[-1] = element # counter
                counter += 1
            
            # generate next array item?
            
            ''' retrieve a batch of sequences of size number requested sequences '''
            prompt_batch = [prompt + json.dumps(selected_object) for prompt, selected_object in zip(self.get_prompt(), selected_objects)]
            prompt_batch = [prompt[:-1] for prompt in prompt_batch]

            self.debug("[generate_array]", prompt_batch[:2])  
            
            ''' tokenize the number of sequences as a batch at this point '''
            encoded_prompts = [self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0).to(self.model.device) for prompt in prompt_batch]
            max_length = max(tensor.size(0) for tensor in encoded_prompts)
            input_tokens = [
                torch.nn.functional.pad(
                    encoded_prompt,
                    (max_length - encoded_prompt.size(0), 0),
                    mode='constant',
                    value=self.tokenizer.pad_token_id
                ) for encoded_prompt in encoded_prompts
            ]
            input_tokens = torch.stack(input_tokens, dim=0)
        
            output = self.model.forward(input_tokens)
            logits = output.logits[:, -1]  # Use all elements along the batch dimension
            
            ''' get the (first) token for comma and closing-square-bracked '''
            binary_token_ids = torch.tensor([
                self.tokenizer.encode(",", add_special_tokens=False)[0],  # Extract the first element
                self.tokenizer.encode("]", add_special_tokens=False)[0]])  # Extract the first element
        
            # Apply temperature and sample
            logits_with_temperature = logits[:, binary_token_ids] / max(self.temperature, 1e-5)
            sampled_tokens = torch.multinomial(torch.nn.functional.softmax(logits_with_temperature, dim=-1), 1)
            selectors = sampled_tokens[:, 0] == 0

            self.debug("[generate_array] - new selectors", selectors)

        return objects


    def remove_closing_quotes(self, input_strings: List[str]):
        # Define a regular expression pattern to match ending quotes excluding escaped quotes
        pattern = re.compile(r'(?<!\\)"')
        
        # Function to remove closing quotes and beyond
        def remove_quotes_and_beyond(input_string):
            match = pattern.search(input_string)
            if match:
                # Found an unescaped ending quote, truncate the string at that position
                return input_string[:match.start()]
            else:
                # No unescaped ending quote found, return the original string
                return input_string

        # Apply the function to each input string
        return [remove_quotes_and_beyond(string) for string in input_strings]


    def get_prompt(self):
        template = """{prompt}\nOutput result in the following JSON schema format:\n{schema}\nResult: {progress}"""

        ''' build prompt for each requested sequence '''
        prompt_batch = []
        
        for i, value in enumerate(self.values):
            
            progress = json.dumps(value)
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

            prompt_batch.append(prompt)

        self.debug("[get_prompt]", prompt_batch[:2])

        return prompt_batch


    def __call__(self) -> Dict[str, Any]:

        ''' set value for each sequence requested '''
        self.values = [{} for _ in range(self.num_return_sequences)]
            
        generated_data = self.generate_object(
            self.json_schema["properties"], self.values
        )
        return generated_data
