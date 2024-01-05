from tqdm import tqdm

from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

from model import EvalModel


# Ref: Instruct_eval: https://github.com/declare-lab/instruct-eval
def entry_point(
    problem_file: str,
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(
        sample_file, k, n_workers, timeout, problem_file
    )

    return results


def filter_code(completion: str, prompt: str = None, template: str = "") -> str:
    if prompt is not None:
        # remove the prompt, since it's a completion model
        completion = completion.replace(prompt, "")
    if template == "md":
        ## Remove boilerplate for the function, reused pure_mode in generation_pipeline
        # select the text between the two '''
        completion = completion.split("'''")[1]
        # remove the first line (which is the language)
        completion = "\n".join(completion.split("\n")[1:])

    ## The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def gen_instruct_prompt(prompt: str):
    """
    Converts the human-eval prompts for completion tasks
    to an instruction prompt to be fed into our instruct-tuned models.
    example:
    from typing import List def all_prefixes(string: str) -> List[str]: \"\"\" {{PROBLEM description}} \"\"\"
    becomes:
    Please complete the following Python code without providing any additional tasks such as testing or explanations
    ### Instruction: Write a Python function 'all_prefixes(string: str) -> List[str]' to solve the following problem:
    :param prompt: the human-eval prompt
    :return: the instruction prompt
    """
    seperator = '"""'  # most of the prompts have this seperator
    if prompt.count(seperator) == 0:
        seperator = "'''"  # some cases use single quotes
    # split the prompt by the triple quotes
    function_signature, description, _ = prompt.split(seperator, 2)
    # get the pure function signature, starts with "def"
    found_def = function_signature.find("def") != -1
    # HumanEval/64 testcase has an extra """ before def so it messes up the split above"
    # if it's a wrong split, try to iterate through the split and find the one that starts with def
    if not found_def:
        splitted_prompt = prompt.split(seperator)
        signeture_indx = -1
        for i, element in enumerate(splitted_prompt):
            if (element.strip().find("def")) != -1:
                signeture_indx = i
                break
        if signeture_indx == -1:
            raise ValueError("Cannot find the function signature")

        function_signature = splitted_prompt[signeture_indx]
        description = splitted_prompt[signeture_indx + 1]

    function_signature = function_signature[function_signature.find("def") : -2].strip()
    description = description.strip()
    # if the prompt doesn't start with "def" right away, it means it had a meanful code before it like imports or specifications
    # so we need to add them to the prompt somehow, we add them as ### Input: using the Alpaca instruction prompt
    if prompt.strip().find("def") != 0:
        description = (
            description + "\n\n### Input:\n" + prompt[: prompt.find("def")].strip()
        )

    # everything is ready now.
    # notice we updated the system message to avoid additional boilerplate code
    system_msg = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\nPlease complete the following Python code without providing any additional tasks such as testing or explanations\n\n"
    prompt_processed = f"### Instruction: Write a Python function '{function_signature}' to solve the following problem:\n{description}\n\n### Output:\n"
    return system_msg + prompt_processed


# def test_gen_instruct_prompt():
#     prompt = 'from typing import List def all_prefixes(string: str) -> List[str]: """ {{PROBLEM description}} """'
#     print(gen_instruct_prompt(prompt))


# Ref: Instruct_eval: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py
def count_indent(text: str) -> int:
    """count the number of leading spaces"""
    count = 0
    for char in text:
        if char == " ":
            count += 1
        else:
            break
    return count


def fix_indents(text: str, multiple: int = 2):
    outputs = []
    for line in text.split("\n"):
        # add spaces to the beginning of the line to make % multiple == 0
        while count_indent(line) % multiple != 0:
            line = " " + line
        outputs.append(line)

    return "\n".join(outputs)


def evaluate(model, data_path: str = None, **kwargs) -> dict:
    dataset = read_problems()
    n_sample = kwargs.get("n_sample", 1)
    best_temperature = {
        1: 0.1,
        10: 0.6,
        100: 0.8,
    }  # best temperature for each n_sample according to instruct_eval
    samples = []
    progress_bar = tqdm(total=len(dataset) * n_sample, desc="Generating samples")
    for task_id in dataset:
        for i in range(n_sample):
            prompt = dataset[task_id]["prompt"]
            prompt = gen_instruct_prompt(prompt)
            temperature = best_temperature[n_sample]
            if temperature > 0:
                completion = model.generate(
                    prompt, temperature=temperature, do_sample=True
                )
            else:
                completion = model.generate(prompt)

            completion = fix_indents(completion)
            sample = dict(task_id=task_id, completion=filter_code(completion))
            if i == 0:
                print("Prompt: ", "-" * 100)
                print(prompt)
                print("Completion: ", "-" * 100)
                print(filter_code(completion))
            samples.append(sample)
            progress_bar.update(1)
    progress_bar.close()

    model_name = model.model_path.replace("/", "_")
    pred_filename = f"humaneval_{model_name}_predictions.jsonl"
    write_jsonl(pred_filename, samples)
    print("Evaluating...")
    result = entry_point(problem_file=data_path, sample_file=pred_filename)
    return result


def test_model_eval():
    # model_path = "parsak/codegen-350M-mono-lora-instruction"
    model_path = "Salesforce/codegen-350M-mono"
    lora_path = "lora-finetunedcodegen-350M-mono-temp3"
    model = EvalModel(model_path=model_path, lora_path=lora_path, device="cpu")

    prompt = 'from typing import List, Tuple def find_closest_elements(numbers: List[float]) -> Tuple[float, float]: """ From a supplied list of numbers (of length at least two) select and return two that are the closest to each other and return them in order (smaller number, larger number). >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2]) (2.0, 2.2) >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0]) (2.0, 2.0) """'
    print(f"Prompt: {prompt}")
    print(f"Instruct prompt: {gen_instruct_prompt(prompt)}")

    output = model.generate(prompt)
    print(f"Normal output:\n{output}")
    print(f"Filtered output:\n{filter_code(output)}")


def test_gen_prompt_from(task_id: str = "HumanEval/64"):
    data_path = "human_eval/data/functional_correctness/valid.jsonl"
    dataset = read_problems()
    # {"task_id": "HumanEval/64", "prompt": "\nFIX = \"\"\"\nAdd more test cases.\n\"\"\"\n\ndef vowels_count(s):\n    \"\"\"Write a ..."}
    prompt = dataset[task_id]["prompt"]
    print(gen_instruct_prompt(prompt))


def test_evaluate():
    model_path = "Salesforce/codegen-350M-mono"
    lora_path = "lora-finetunedcodegen-350M-mono-temp3"
    model = EvalModel(model_path=model_path, lora_path=lora_path, device="cpu")

    data_path = "humaneval_Salesforce_codegen-350M-mono_predictions.jsonl"
    result = evaluate(model, data_path, n_sample=1)


if __name__ == "__main__":
    # test_gen_prompt_from("HumanEval/64")
    test_evaluate()
