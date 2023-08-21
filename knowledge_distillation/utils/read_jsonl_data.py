import ast
import re
from pathlib import Path
from typing import List


def read_jsonl_data(jsonl_path: Path) -> List[dict]:
    data_string = jsonl_path.read_text().strip()

    dialogs_dicts: List[dict] = list()
    for dialog_str in re.split(r"\}\n?\{", data_string):
        if dialog_str[0] != "{":
            dialog_str = "{" + dialog_str
        if dialog_str[-1] != "}":
            dialog_str = dialog_str + "}"
        try:
            dialogs_dicts.append(ast.literal_eval(dialog_str))
        except Exception as exception:
            print(dialog_str)
            print(exception)
            break

    return dialogs_dicts
