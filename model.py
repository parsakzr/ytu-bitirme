from peft import PeftModel
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from pydantic import BaseModel, ConfigDict
from typing import Optional, Tuple


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    model_path: str
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False
    max_input_length: int = 512
    max_output_length: int = 512

    model_config = ConfigDict(protected_namespaces=())

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load()

    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **args)
            if self.lora_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
                self.model = self.model.merge_and_unload()
            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def generate(
        self, prompt: str, verbose: bool = False, pure_mode: bool = True, **kwargs
    ):
        def generate_prompt(text):
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            prompt += f"### Instruction: {text}\n\n### Output:\n"
            return prompt

        prompt = generate_prompt(prompt)
        if verbose:
            print(f"------------ Prompt -------------\n{prompt}")

        input_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        ).input_ids
        if not self.load_8bit:
            input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # avoid warning
            **kwargs,
            # no_repeat_ngram_size=1,
            # early_stopping=True,
            # num_beams=2,
            # temperature=0.1,
            # do_sample=True,
        )

        output = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if pure_mode:
            # remove the prompt, since it's a completion model
            output = output.replace(prompt, "")
            # select the text between the two '''
            output = output.split("'''")[1]
            # remove the first line (which is the language)
            output = "\n".join(output.split("\n")[1:])
        if verbose:
            print(f"-------- Generated Output --------\n{output}")

        return output


def test_Model():
    model_id = "Salesforce/codegen-350M-mono"
    lora_id = "lora-finetunedcodegen-350M-mono-temp3"
    model = EvalModel(model_path=model_id, lora_path=lora_id, device="cpu")

    prompt = "Create a function to print Hello world!"

    output = model.generate(prompt, verbose=True, temperature=0.5, do_sample=True)

    print(output)


if __name__ == "__main__":
    test_Model()
