def simple_func(a, b):
    return a + b

def with_type_hints(name: str, age: int) -> str:
    return f"{name} is {age}"

def with_defaults(x: int = 0, y: str = "hello"):
    pass
