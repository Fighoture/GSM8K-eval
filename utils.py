import gzip
import json

from transformers import StoppingCriteria

class StopOnKeyword(StoppingCriteria):
    def __init__(self, tokenizer, stop_string, existing_number=1):
        self.tokenizer = tokenizer
        self.stop_string = stop_string
        self.existing_number = existing_number

    def __call__(self, input_ids, scores, **kwargs):
        for i in range(input_ids.shape[0]):
            text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            stop_string_occurrences = len(text.split(self.stop_string)) - 1
            if stop_string_occurrences <= self.existing_number:
                return False
        return True


def load_jsonl(
    file_path,
    instruction="instruction",
    input="input",
    output="output",
    category="category",
    is_gzip=False,
):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    open_func = open if not is_gzip else gzip.open
    with open_func(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None,
            )
            item = new_item
            list_data_dict.append(item)
    return list_data_dict
