from jsonschema import validate

simple_weather_schema = {
    "type": "object",
    "properties": {
        "humidity": {"type": "number"},
        "temperatureC": {
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["value", "unit"],
        },
    },
    "required": ["humidity", "temperature"],
}

validate(
    {
        "humidity": 0.9,
        "temperature": {
            "value": 37,
        },
    },
    simple_weather_schema,
)
