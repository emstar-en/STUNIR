from typing import List, Optional

@dataclass
class Person:
    name: str

def simple_func(a, b):
    return a + b

def with_type_hints(name: str, age: int) -> str:
    return f"{name} is {age}"

def with_defaults(x: int = 0, y: str = "hello"):
    pass

def multiline_params(
    first: str,
    second: int,
    third: List[str]
):
    pass

@decorator
@another_decorator
def decorated(a: int) -> int:
    return a * 2
