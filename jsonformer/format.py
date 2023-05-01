from termcolor import colored


def highlight_values(value):
    def recursive_print(obj, indent=0, is_last_element=True):
        if isinstance(obj, dict):
            print("{")
            last_key = list(obj.keys())[-1]
            for key, value in obj.items():
                print(f"{' ' * (indent + 2)}{key}: ", end="")
                recursive_print(value, indent + 2, key == last_key)
            print(f"{' ' * indent}}}", end=",\n" if not is_last_element else "\n")
        elif isinstance(obj, list):
            print("[")
            for index, value in enumerate(obj):
                print(f"{' ' * (indent + 2)}", end="")
                recursive_print(value, indent + 2, index == len(obj) - 1)
            print(f"{' ' * indent}]", end=",\n" if not is_last_element else "\n")
        else:
            if isinstance(obj, str):
                obj = f'"{obj}"'
            print(colored(obj, "green"), end=",\n" if not is_last_element else "\n")

    recursive_print(value)
