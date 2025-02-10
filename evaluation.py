import os
import json
import argparse

from tqdm.auto import tqdm

from utils.llms_interface import LanguageModelInterface


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.model = LanguageModelInterface(self.args)
        self.label_phrase = 'The correct option is:'
        
    def load_in_context_examples(self):
        if self.args.dataset_name == 'ProverGen':
            with open(os.path.join(self.args.demonstration_path, f"{self.args.dataset_name}.json")) as f:
                example_dict = json.load(f)
            in_context_examples = example_dict[f"{self.args.split}_{self.args.mode}"]
        else:
            with open(os.path.join(self.args.demonstration_path, f'{self.args.dataset_name}.json'), 'r') as f:
                in_context_examples = json.load(f)
            in_context_examples = in_context_examples[self.args.mode]
            
        return in_context_examples
    
    def load_raw_dataset(self, split):
        with open(os.path.join(self.args.data_path, self.args.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
            
        return raw_dataset
    
    def create_prompt(self, in_context_example, test_example):
        full_prompt = in_context_example
        context = test_example['context'].strip()
        question = test_example['question'].strip()
        options = '\n'.join([opt.strip() for opt in test_example['options']])
        full_prompt = full_prompt.replace('[[CONTEXT]]', context)
        full_prompt = full_prompt.replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[OPTIONS]]', options)
        
        return full_prompt
    
    def evaluate(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.args.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.args.split} split.")
        
        # load in-context examples
        if self.args.trained_model:
            in_context_examples = "Context:\n[[CONTEXT]]\n\nQuestion: [[QUESTION]]\n\nOptions:\n[[OPTIONS]]\n\nThe correct option is:"
        else:
            in_context_examples = self.load_in_context_examples()
        
        outputs = []
        cnt = -1
        for example in tqdm(raw_dataset):
            cnt += 1
            if cnt < self.args.start or cnt >= self.args.end:
                continue
            
            question = example['question']

            # create prompt
            full_prompt = self.create_prompt(in_context_example=in_context_examples, test_example=example)
            
            if self.args.mode == 'CoT':
                full_prompt = [
                    {'role': 'system', 'content': "Given a problem statement as contexts, the task is to answer a logical reasoning question. Your answer should be in JSON format with keys: reasoning, answer."},
                    {'role': 'user', 'content': full_prompt}
                ]
            else:
                full_prompt = [
                    {'role': 'system', 'content': "Given a problem statement as contexts, the task is to answer a logical reasoning question. Your answer should be in JSON format with key: answer."},
                    {'role': 'user', 'content': full_prompt}
                ]
            # print(full_prompt)
            result = self.model.completion(full_prompt).response_text
            
            if self.args.verbose:
                print(result)
            
            # create output
            output = {'id': example['id'],
                      'context': full_prompt,
                      'question': question,
                      'label': example['answer'],
                      'model_answer': result
                    }
            outputs.append(output)
        
        model_name = self.args.model_name.split('/')[-1]
        # save outputs        
        with open(os.path.join(self.args.output_dir, f'{self.args.mode}_{self.args.dataset_name}_{self.args.split}_{model_name}_{self.args.start}-{self.args.end}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument('--base_url', type=str, default="localhost:6417")
    parser.add_argument('--api_key', type=str, default="EMPTY")
    parser.add_argument('--dataset_name', type=str, default='ProverGen')
    parser.add_argument('--split', type=str, default='hard')
    parser.add_argument('--mode', type=str, default='CoT')  # Direct, CoT
    parser.add_argument("--output_dir", type=str, default='result/')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=600)
    parser.add_argument('--trained_model', action="store_true")
    
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--data_path', type=str, default='logic_data/')
    parser.add_argument('--demonstration_path', type=str, default='logic_data/icl_examples')
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    if args.mode == "Direct":
        args.max_new_tokens = 128
    else:
        args.max_new_tokens = 1024
    
    evaluator = Evaluator(args)
    evaluator.evaluate()

    
