import json
import os
import random
import re
import time
from functools import lru_cache

import numpy as np
import openai


def load_data(args):
    random.seed(args.seed)
    data_root = 'dizoo/tabmwp/data'
    problems = json.load(open(os.path.join(data_root, f'problems_train.json')))

    pids = list(problems.keys())

    samples = random.sample(pids, args.train_number + args.cand_number)  # random sample
    train_pids = samples[:args.train_number]
    cand_pids = samples[args.train_number:]
    return problems, cand_pids, train_pids


def get_gpt3_output(prompt, args):
    return call_gpt3(args.engine, prompt, args.temperature, args.max_tokens, args.top_p, args.frequency_penalty,
                     args.presence_penalty)


@lru_cache(maxsize=10000)
def call_gpt3(engine, prompt, temperature, max_tokens, top_p, frequency_penalty, presence_penalty):
    patience = 100
    while True:
        try:
            response = openai.Completion.create(engine=engine,
                                                prompt=prompt,
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                top_p=top_p,
                                                frequency_penalty=frequency_penalty,
                                                presence_penalty=presence_penalty,
                                                stop=["\n"])
            output = response["choices"][0]["text"].strip()
            break
        except Exception as e:
            patience -= 1
            if not patience:
                print("!!! running out of patience waiting for OpenAI")
            else:
                time.sleep(0.1)
    return output


def get_table_text(problem):
    table = problem['table']
    title = problem['table_title']
    if title and len(title) > 0:
        table = f"[TITLE]: {title}\n{table}"
    return table


def get_question_text(problem, option_inds):
    question = problem['question']

    unit = problem['unit']
    if unit and len(unit) > 0:
        question = f"{question} (Unit: {unit})"

    choices = problem['choices']
    if choices and len(choices) > 0:
        choice_list = []
        for i, c in enumerate(choices):
            choice_list.append("({}) {}".format(option_inds[i], c))
        options = " ".join(choice_list)
        question = f"{question}\nOptions: {options}"

    return question


def get_answer(problem):
    return problem['answer']


def get_solution_text(problem):
    # \\n: GPT-3 can generate the solution with more tokens
    solution = problem['solution'].replace("\n", "\\n")
    return solution


def create_one_example(format, table, question, answer, solution, test_example=True):

    input_format, output_format = format.split("-")  # e.g., "TQ-A"

    elements = {
        "Q": f"Question: {question}",
        "T": f"Table: {table}",
        "S": f"Solution: {solution}",
        "A": f"Answer: The answer is {answer}.",
        "AS": f"Answer: The answer is {answer}. BECAUSE: {solution}",
        "SA": f"Answer: {solution} The answer is {answer}."
    }

    # Input
    input = "\n".join(elements[label] for label in input_format)

    # Output
    if test_example:
        output = "Answer:"
    else:
        output = elements[output_format]

    # Prompt text
    text = input + "\n" + output
    text = text.replace("  ", " ").strip()

    return text


def build_prompt(problems, shot_pids, test_pid, args):

    examples = []
    pids = shot_pids + [test_pid]

    # n-shot training examples
    for pid in pids:
        problem = problems[pid]
        table = get_table_text(problem)
        question = get_question_text(problem, args.option_inds)
        answer = get_answer(problem)
        solution = get_solution_text(problems[pid])

        if pid == test_pid:
            assert pid not in shot_pids
            example = create_one_example(args.prompt_format, table, question, answer, solution, test_example=True)
        else:
            example = create_one_example(args.prompt_format, table, question, answer, solution, test_example=False)

        examples.append(example)

    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input


def extract_prediction(output, options, option_inds):
    # $\\frac{16}{95}$ -> 16/95
    output = re.sub(r"\$?\\frac\{([\d\.\,\-]+)\}\{([\d\.\,]+)\}\$?", r"\1/\2", output)

    output = re.sub(r"(?<![AP]\.M)\.$", "", output)
    output = re.sub(r"(?<=\d)[\=](?=[\-\$\d])", " = ", output)
    output = re.sub(r"\u2212", "-", output)

    ## Multi-choice questions
    if options:
        patterns = [
            r'^\(([A-Za-z])\)$',  # "(b)", "(B)"
            r'^([A-Za-z])$',  # "b", "B"
            r'^([A-Za-z]). ',  # "b", "B"
            r'[Th]he answer is ([A-Z])',  # "The answer is B"
            r'^\(([A-Za-z])\) [\s\S]+$',  # "(A) XXXXX"
            r'[Th]he answer is \(([A-Za-z])\) [\s\S]+$',  # "The answer is (B) XXXXX."
        ]

        # have "X" in the output
        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                pred = res[0].upper()  # e.g., "B"
                if pred in option_inds:
                    ind = option_inds.index(pred)  # 1
                    if ind >= len(options):
                        ind = random.choice(range(len(options)))
                    predition = options[ind]
                    return predition

        # find the most similar options
        scores = [score_string_similarity(x, output) for x in options]
        max_idx = int(np.argmax(scores))  # json does not recognize NumPy data types
        predition = options[max_idx]
        return predition

    else:
        ## free_text QA problems, numeric answer
        patterns = [
            # r'^\([A-Za-z]\) ([\s\S]+)$', # "(A) XXXXX"
            # r'[Th]he answer is \([A-Za-z]\) ([\s\S]+)$', # "The answer is (B) XXXXX."
            r'[Th]he answer is ([\s\S]+)$',  # "The answer is XXXXX.",
            r'[Th]he table shows that ([\d\$\.\,\/\:]+) ',
            r' = ([\d\$\.\,\/\:]+)',  # "= $1.40"
            r'(?<= be| is) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "will be $1.40"
            r'(?<= are| was) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "are $1.40"
            r'(?<= were) ([\-\d\$\.\,\/\:]{0,}[\d]+)',  # "are $1.40"
            r' ([\d\$\.\,\/\:]+ [AP]\.M\.)',  # 7:25 P.M.
            r'([\-\d\$\.\,\/\:]{0,}[\d]+)',  # 14.5
        ]

        for p in patterns:
            pattern = re.compile(p)
            res = pattern.findall(output)
            if len(res) > 0:
                predition = res[-1].strip()
                if predition.endswith(".") and ".M." not in predition:
                    predition = predition[:-1]
                return predition

    return output


def normalize_answer(text, unit):
    # ["1,000", "123", "3/4", "56.456", "$56.4", "-3", "-10.02", "-3/2"]

    text = re.sub("^[\$]", "", text)
    text = re.sub("[\,\.\,\/]$", "", text)
    result = re.match("^[-+]?[\d,./]+$", text)

    if result is not None:
        # is number?
        text = text.replace(",", "")
        result = re.match("[-+]?\d+$", text)
        try:
            if result is not None:
                number = int(text)
            elif "/" in text:
                nums = text.split("/")
                number = round(float(nums[0]) / float(nums[1]), 3)
            else:
                number = round(float(text), 3)
            number = str(number)
            number = re.sub(r"\.[0]+$", "", number)
            return number
        except:
            return text
    else:
        # is text
        if unit:
            text = text.replace(unit, "").strip()
        return text


def score_string_similarity(str1, str2):
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def create_example_from_pid(pid, problems, args, test=False):
    problem = problems[pid]
    table = get_table_text(problem)
    question = get_question_text(problem, args.option_inds)
    answer = get_answer(problem)
    solution = get_solution_text(problems[pid])

    if test:
        example = create_one_example(args.prompt_format, table, question, answer, solution, test_example=True)
    else:
        example = create_one_example(args.prompt_format, table, question, answer, solution, test_example=False)

    return example
