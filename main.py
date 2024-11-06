import torch
import re
import os
import random
import json
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig, StoppingCriteria
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from utils import load_jsonl, StopOnKeyword
import argparse

transformers.logging.set_verbosity(40)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
ANSWER_TRIGGER = "The answer is"
MAX_MODEL_LENGTH = 4096
MAX_NEW_TOKEN = 256

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def create_demo_text(n_shot=8, cot_flag=True):
    question, chain, answer = [], [], []
    question.append(
        "There are 15 trees in the grove. "
        "Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. "
        "How many trees did the grove workers plant today?"
    )
    chain.append(
        "There are 15 trees originally. "
        "Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6."
    )
    answer.append("6")

    question.append(
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?"
    )
    chain.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
    answer.append("5")

    question.append(
        "Leah had 32 chocolates and her sister had 42. If they ate 35, "
        "how many pieces do they have left in total?"
    )
    chain.append(
        "Originally, Leah had 32 chocolates. "
        "Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39."
    )
    answer.append("39")

    question.append(
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?"
    )
    chain.append(
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8."
    )
    answer.append("8")

    question.append(
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?"
    )
    chain.append(
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9."
    )
    answer.append("9")

    question.append(
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. "
        "How many computers are now in the server room?"
    )
    chain.append(
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 is 29."
    )
    answer.append("29")

    question.append(
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?"
    )
    chain.append(
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, "
        "he had 35 - 2 = 33 golf balls."
    )
    answer.append("33")

    question.append(
        "Olivia has $23. She bought five bagels for $3 each. "
        "How much money does she have left?"
    )
    chain.append(
        "Olivia had 23 dollars. "
        "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8."
    )
    answer.append("8")

    # randomize order of the examples ...
    index_list = list(range(len(question)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    demo_text = ""
    for i in index_list[:n_shot]:
        if cot_flag:
            demo_text += (
                "Q: "
                + question[i]
                + "\nA: "
                + chain[i]
                + " "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
        else:
            demo_text += (
                "Question: "
                + question[i]
                + "\nAnswer: "
                + ANSWER_TRIGGER
                + " "
                + answer[i]
                + ".\n\n"
            )
    return demo_text


def build_prompt(input_text, n_shot, cot_flag):
    demo = create_demo_text(n_shot, cot_flag)
    input_text_prompt = demo + "Q: " + input_text + "\nA:"
    return input_text_prompt


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred


def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_peft_model(model, peft_model_path):
    model = PeftModel.from_pretrained(model, peft_model_path)
    model = model.merge_and_unload()
    return model


def load_model(model_path, prune_result, peft_model):
    print("transformers model loading")
    if prune_result != ".":
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.pre_ffn_hidden = True

        if prune_result.endswith(".json"):
            pruning_file_path = prune_result
        elif "c4_prune.json" in os.listdir(prune_result):
            pruning_file_path = f"{prune_result}/c4_prune.json"
        elif "MathInstruct_prune.json" in os.listdir(prune_result):
            pruning_file_path = f"{prune_result}/MathInstruct_prune.json"
        else:
            raise FileNotFoundError("Could not find pruning file.")
        print(f"pruning_file_path: {pruning_file_path}")
        with open(pruning_file_path, "r") as f:
            pruned_mask = json.load(f)
        config.pruned_mask = pruned_mask
        config.max_position_embeddings = MAX_MODEL_LENGTH

        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
        )

    llm.generation_config = GenerationConfig.from_pretrained(model_path)
    llm.generation_config.pad_token_id = llm.generation_config.eos_token_id
    if peft_model != ".":
        dir_list = os.listdir(peft_model)
        if "adapter_model.safetensors" not in dir_list:
            for dir_path in dir_list:
                peft_model_path = os.path.join(peft_model, dir_path)
                llm = load_peft_model(llm, peft_model_path)
                print(f"model peft merge from {peft_model_path}")
        else:
            llm = load_peft_model(llm, peft_model)
            print(f"model peft merge from {peft_model}")

    print(f"transformers model loading finish.")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("tokenizer loaded.")
    return llm, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/dataset/llama2/llama-2-7b-hf",
        help="The model path.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="The dir of the data.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data",
        help="The dir of saving.",
    )
    parser.add_argument("--peft_model", "-pm", type=str, default=".")
    parser.add_argument("--prune_result", "-pr", type=str, default=".")
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--seed", "-s", type=int, default=1993)

    args = parser.parse_args()
    return args


def generate(model, tokenizer, batch_inputs, stop_criteria, generate_kwargs):
    input_text = tokenizer(batch_inputs, padding=True, return_tensors="pt")

    output_ids = model.generate(**input_text, **generate_kwargs, stopping_criteria=[stop_criteria])
    response = []
    for i in range(output_ids.shape[0]):
        response.append(
            tokenizer.decode(output_ids[i][len(input_text["input_ids"][i]):], skip_special_tokens=True, ignore_tokenization_space=True)
        )

    if len(response) > 1:
        return response
    return response[0]


def args_generate_path(args):
    model_name = args.model_path.split("/")[-1]
    if "/result/" in args.peft_model:
        save_pre_list = args.peft_model.split("/result/")[-1].split("/")
    elif "/pruned_result/" in args.prune_result:
        save_pre_str = args.prune_result.split("/pruned_result/")[-1]
        if save_pre_str.endswith(".json"):
            flag = save_pre_str.split("/")[-1].split(".json")[0]
            save_pre_list = save_pre_str.split("/")[:-1]
            save_pre_list = ["pruned"] + save_pre_list + [flag]
        else:
            save_pre_list = save_pre_str.split("/")
            save_pre_list = ["pruned"] + save_pre_list
    else:
        save_pre_list = []
    return [model_name] + save_pre_list


def check_exist(exists_result, sample):
    for each in exists_result:
        if sample["instruction"] == each["instruction"]:
            if "generation" in each:
                return True
            else:
                return False
        else:
            continue
    return False


def main():
    seed_everything(args.seed)

    test_filepath = os.path.join(args.data_dir, "test.jsonl")
    list_data_dict = load_jsonl(test_filepath, instruction="question", output="answer")

    output_dir = os.path.join(args.save_dir, "/".join(args_generate_path(args)))
    print("output_dir: ", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "results.txt")
    if os.path.exists(output_path):
        with open(output_path, "r") as fi:
            exists_result = json.load(fi)
    else:
        exists_result = []

    prompt_inputs = []
    outputs = []
    for sample in tqdm(list_data_dict):
        if check_exist(exists_result, sample):
            continue

        input_text = build_prompt(sample["instruction"], N_SHOT, COT_FLAG)
        prompt_inputs.append(input_text)
        outputs.append(sample["output"])

    model, tokenizer = load_model(args.model_path, args.prune_result, args.peft_model)

    print("processing sample number: ", len(prompt_inputs))
    stop_criteria = StopOnKeyword(tokenizer=tokenizer, stop_string="Q:", existing_number=1 + N_SHOT)
    answers = []
    for batch_idx in tqdm(range(len(prompt_inputs) // args.batch_size)):
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, len(prompt_inputs))
        batch_inputs = prompt_inputs[start_idx:end_idx]
        batch_outputs = outputs[start_idx:end_idx]
        generate_kwargs = dict(max_new_tokens=MAX_NEW_TOKEN, top_p=0.95, temperature=0.8)

        model_completion = generate(model, tokenizer, batch_inputs, stop_criteria, generate_kwargs)
        model_answer = [clean_answer(completion) for completion in model_completion]

        for i in range(args.batch_size):
            original_input = batch_inputs[i].split("Q: ")[-1].split("\nA:")[0]
            element = {"instruction": original_input, "output": batch_outputs[i], "generation": model_completion[i]}
            exists_result.append(element)
        with open(output_path, "w") as f:
            f.write(json.dumps(exists_result))

        is_cor = [is_correct(model_answer[i], batch_outputs[i]) for i in range(len(model_answer))]
        answers.extend(is_cor)
        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers)) / len(answers)}."
        )


    with open(os.path.join(output_dir, "scores.txt"), "w") as f:
        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers))/len(answers)}.",
            file=f,
        )


if __name__ == "__main__":
    args = parse_args()
    main()
