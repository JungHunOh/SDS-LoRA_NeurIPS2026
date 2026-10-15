''' GSM8K and MATH accuracy with greedy decoding (MetaMath protocol). '''
import os
import re
import sys
import time
from datetime import timedelta
from fractions import Fraction

import jsonlines
import torch
from tqdm import tqdm

from . import math_util as util

MAX_INT = sys.maxsize
INVALID_ANS = "[invalid]"

invalid_outputs = []


def evaluate(model, tokenizer, cfg, output_dir):
    e = cfg['eval']
    test_gsm8k(model, tokenizer, e['gsm8k_file'], os.path.join(output_dir, 'gsm8k_eval.txt'), batch_size=e['batch_size'])
    test_hendrycks_math(model, tokenizer, e['math_file'], os.path.join(output_dir, 'math_eval.txt'), batch_size=e['batch_size'])


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) and is_number(numerator):
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        return round(float(frac.numerator / frac.denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])
    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def process_results(doc, completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp)>0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if util.is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False


def test_gsm8k(model, tokenizer, data_path, save_path, start=0, end=MAX_INT, batch_size=100):
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('prompt template:', problem_prompt)

    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["query"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['response'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print('length of inputs:', len(gsm8k_ins))

    model.to("cuda")
    model.eval()

    res_completions = []
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    total_batches = len(batch_gsm8k_ins)
    start_time = time.time()

    for batch_idx, batch_prompts in enumerate(batch_gsm8k_ins):
        # Tokenize and move inputs to device
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, output in enumerate(outputs):
            gen_text = tokenizer.decode(output, skip_special_tokens=True)
            res_completions.append(gen_text)

        # Added: Print progress and estimated finish time
        elapsed = time.time() - start_time
        batches_left = total_batches - (batch_idx + 1)
        est_remaining = (elapsed / (batch_idx + 1)) * batches_left
        est_finish = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + est_remaining))
        print(f"[Batch {batch_idx+1}/{total_batches}] Elapsed: {timedelta(seconds=int(elapsed))}, Estimated finish: {est_finish}")

    invalid_outputs = []
    result = []
    for prompt, completion, prompt_answer in zip(gsm8k_ins, res_completions, gsm8k_answers):
        y_pred = extract_answer_number(completion)
        if y_pred is not None:
            result.append(float(y_pred) == float(prompt_answer))
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)

    correct_count = sum(result)
    acc = correct_count / len(result)
    
    # Added: Final summary print
    print("\nFinal results:")
    print(f"Number of correct answers: {correct_count}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Number of invalid outputs: {len(invalid_outputs)}")

    # Added: Save results to file
    with open(save_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"Number of correct answers: {correct_count}\n")
        f_out.write(f"Accuracy: {acc:.4f}\n")
        f_out.write(f"Number of invalid outputs: {len(invalid_outputs)}\n")
    print(f"Results saved to {save_path}")


def test_hendrycks_math(model, tokenizer, data_path, save_path, start=0, end=MAX_INT, batch_size=200):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    print('prompt template:', problem_prompt)
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            solution = item['output']
            temp_ans = remove_boxed(util.last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    print('total length ===', len(hendrycks_math_ins))
    hendrycks_math_ins = hendrycks_math_ins[start:end]
    hendrycks_math_answers = hendrycks_math_answers[start:end]
    print('length of inputs:', len(hendrycks_math_ins))
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=batch_size)


    model.to("cuda")
    model.eval()

    res_completions = []
    for prompt in tqdm(batch_hendrycks_math_ins, desc="Generating completions"):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            completions = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for output in completions:
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            res_completions.append(generated_text)

    results = []
    for idx, (prompt, completion, prompt_answer) in enumerate(zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)):
        res = process_results(prompt, completion, prompt_answer)
        results.append(res)

    correct_count = sum(results)
    acc = correct_count / len(results)

    print("\nFinal results:")
    print(f"Number of correct answers: {correct_count} / {len(results)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Number of invalid outputs: {len(invalid_outputs)}")

    with open(save_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"Number of correct answers: {correct_count} / {len(results)}\n")
        f_out.write(f"Accuracy: {acc:.4f}\n")
        f_out.write(f"Number of invalid outputs: {len(invalid_outputs)}\n")
    print(f"Results saved to {save_path}")
