''' Generation-based accuracy on the commonsense benchmarks (LLM-Adapters protocol). '''
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def evaluate(model, tokenizer, cfg, output_dir):
    data_dir, batch_size = cfg['eval']['data_dir'], cfg['eval']['batch_size']
    result_file = os.path.join(output_dir, 'commonsense_eval.txt')

    def generate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=32,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=inputs['attention_mask'].to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[1].strip() for o in outputs]
        return outputs

    model.eval()

    accs = []

    for ds in cfg['eval']['datasets']:
        dataset = load_data(data_dir, ds)
        batches = create_batch(dataset, batch_size)
        total = len(batches)
        correct = 0
        current = 0
        pbar = tqdm(total=total)
        for idx, batch in enumerate(batches):
            current += len(batch)
            instructions = [data.get('instruction') for data in batch]

            outputs = generate(instructions)

            for data, output in zip(batch, outputs):
                label = data.get('answer')
                predict = extract_answer(ds, output)
                if label == predict:
                    correct += 1
            print('---------------')
            print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
            print('---------------')
            pbar.update(1)
        pbar.close()
        with open(result_file, 'a+') as f:
            f.write(f'\rTask:{ds} test:{idx + 1}/{total} | accuracy {correct}  {correct / (current)}')
            accs.append(correct / current)
        print('\n')
        print('test finished')
    with open(result_file, 'a+') as f:
        f.write(f'\rAverage accuracy: {sum(accs) / len(accs)}\n')


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(data_dir, dataset) -> list:
    file_path = os.path.join(data_dir, dataset, 'test.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def extract_answer(dataset, sentence: str) -> float:
    dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
