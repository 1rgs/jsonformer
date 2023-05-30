from jsonformer.format import highlight_values

from jsonformer.main_no_tokens import JsonformerNoTokens
from jsonformer.openai import OpenAIModel

car = {
    "type": "object",
    "properties": {
        "car": {
            "type": "object",
            "properties": {
                "make": {"type": "string"},
                "model": {"type": "string"},
                "year": {"type": "number"},
                "colors": {"type": "array", "items": {"type": "string"}},
                "features": {
                    "type": "object",
                    "properties": {
                        "audio": {
                            "type": "object",
                            "properties": {
                                "brand": {"type": "string"},
                                "speakers": {"type": "number"},
                                "hasBluetooth": {"type": "boolean"},
                            },
                        },
                        "safety": {
                            "type": "object",
                            "properties": {
                                "airbags": {"type": "number"},
                                "parkingSensors": {"type": "boolean"},
                                "laneAssist": {"type": "boolean"},
                            },
                        },
                        "performance": {
                            "type": "object",
                            "properties": {
                                "engine": {"type": "string"},
                                "horsepower": {"type": "number"},
                                "topSpeed": {"type": "number"},
                            },
                        },
                    },
                },
            },
        },
        "owner": {
            "type": "object",
            "properties": {
                "firstName": {"type": "string"},
                "lastName": {"type": "string"},
                "age": {"type": "number"},
            },
        },
    },
}

builder = JsonformerNoTokens(
    model=OpenAIModel("text-curie-001", debug=False),
    json_schema=car,
    prompt="Generate a car object with the following schema:",
    temperature=0.9,
    debug=True,
)

print("Generating...")
output = builder()

highlight_values(output)
