from jsonformer.format import highlight_values
from jsonformer.main import Jsonformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


weather_schema = {
    "type": "object",
    "properties": {
        "temperature": {"type": "number"},
        "humidity": {
            "type": "number",
        },
        "wind_speed": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "unit": {"type": "string"},
            },
        },
    },
}

print("Loading model and tokenizer...")
model_name = "databricks/dolly-v2-12b"

model = AutoModelForCausalLM.from_pretrained(
    model_name, use_cache=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_cache=True)
print("Loaded model and tokenizer")


builder = Jsonformer(
    model=model,
    tokenizer=tokenizer,
    json_schema=weather_schema,
    prompt="generate the weather",
    debug=True,
    device="cuda",
)

print("Generating...")
output = builder()

highlight_values(
    output,
)
