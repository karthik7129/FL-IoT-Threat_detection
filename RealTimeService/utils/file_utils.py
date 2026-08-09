import json
import os
from typing import Any

def read_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)

def write_json(path: str, data: Any) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2)
