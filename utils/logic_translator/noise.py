import re
import time
import json
import random

from typing import Dict, List, Tuple

from tqdm.auto import tqdm
from openai import OpenAI


class NoiseTranslator:
    def __init__(self, args, translated_data) -> None:
        self.args = args
        self.model_name = args.model_name
        self.data = translated_data
        self.examples = self.load_json(self.args.example_path)
        
        if args.api_key == "EMPTY" and args.base_url == "EMPTY":
            self.client = OpenAI()
        else:
            self.client = OpenAI(
                api_key=args.api_key,
                base_url=args.base_url
            )
        self.err_cnt = 0
    
    def create_distracting_rules(self) -> List:
        pbar = tqdm(range(len(self.data)))
        for item in self.data:
            translated_facts = item['translated_facts']
            translated_rules = item['translated_rules']
            name = item['name']
            background_story = item['background']
            name_category = item['name_category']
            
            for rule in translated_rules:
                if "distracting_rule" not in rule.keys():
                    continue
                d_rule = rule["distracting_rule"]
                rule["distracting_rule"] = self.__translate_distracting_rule(rule=d_rule, translated_facts=translated_facts, selected_name=name, background_story=background_story, name_category=name_category)
                
            pbar.update(1)
        
        return self.data
        
        
    def __translate_distracting_rule(self, rule: Dict, translated_facts: Dict, selected_name: str, background_story: str, name_category: str) -> Tuple:
        """
        1. Get forbiden list (avoid using the same predicates)
        2. Transform rules and get references
        3. Translate rules and check whether the predicates is unique
        4. Get the translation result
        """
        rule_expression = rule['expression']
        rule_facts = self.extract_facts(rule_expression)
        rule_fact_placeholder = []
        existed_predicate = []
        
        # 1: Get forbidden list
        # get forbiden list
        forbidden_list = []
        for key in translated_facts.keys():
            forbidden_list.append(translated_facts[key]['expression'])
        
        # 2: Transform rules and get references
        # transform rules, i.e. add ∀x and (x)
        for item in rule_facts:
            if item in translated_facts.keys():
                rule_expression = rule_expression.replace(f"[{item}]", f"{translated_facts[item]['expression']}(x)")
                existed_predicate.append(item)
            else:
                rule_expression = rule_expression.replace(f"[{item}]", f"{item}(x)")
                rule_fact_placeholder.append(item)
        rule_expression = f"∀x ({rule_expression})"
        rule_fact_placeholder.append("universal_rule")
        rule_fact_placeholder.append("specific_rule")
        
        # get references
        assert len(existed_predicate) == 1
        references = ""
        for item in existed_predicate:
            references += f"{translated_facts[item]['expression']}\n{translated_facts[item]['rule']}: {translated_facts[item]['rule_nl']}\n"
        references = references[:-1]
        
        # 3: Translate rules and check whether the predicates is unique
        # get few shot examples
        example_index = ""
        for chr in rule_expression:
            if chr in ["→", "⊕", "∨", "∧"]:
                example_index += chr
        
        few_shot_examples = self.examples['normal'][str(example_index)]
        
        # construct prompt
        translate_logic_prompt = [
            {"role": "system", "content": f"You will be provided a logic expression, a reference of the existed predicate in the logic expression, and a background story. Your task is to replace the placeholders in the logic expression with appropriate predicates (no more than 5 words) so that the provided logic expression represent a real world common sense rule.\n\nDo not use 'not'. Do not use the words in the forbidden list. The words that are similar in meaning to the words in the forbidden list are also not allowed.\n\nYour answer is not required to closely connected to the background story. You can use any predicates you like as long as their length is less than 5.\n\nYour answer should be in JSON format with the provided keys."}
        ]
        translate_logic_prompt.extend(few_shot_examples)
        translate_logic_prompt.append({"role": "user", "content": f"background story:\n{background_story}\n\nreference: {references}\nforbidden list: {forbidden_list}\n\nlogic expression: {rule_expression}\nkeys: {rule_fact_placeholder}\nNote: x belongs to {name_category}."})
        
        # check whether there is an answer in forbidden list
        has_repetition = True
        while has_repetition:
            has_repetition = False
            result = self.__send_request(message=translate_logic_prompt)
            
            for key in result.keys():
                if key in ["universal_rule", "specific_rule"]:
                    continue
                if result[key] in forbidden_list or re.findall(r'^F\d+$', result[key]) != []:
                    has_repetition = True
                    break
        
        # Get the translation result
        # get rule expression and new fact
        for key in result.keys():
            if key in ["universal_rule", "specific_rule"]:
                continue
            result[key] = result[key].replace(" ", "_")
            rule_expression = rule_expression.replace(key, result[key])
            
        specific_rule_expression = rule_expression.replace(f"∀x (", "")[:-1]
        specific_rule_expression = specific_rule_expression.replace("(x)", f"({selected_name})")
        
        distracting_facts = []
        distracting_facts_fol = []
        for key in result.keys():
            if key in ["universal_rule", "specific_rule"]:
                continue
            
            if str(rule_facts.index(key)) not in rule['value'].keys():
                continue
            
            fact_value = rule['value'][str(rule_facts.index(key))]
            if fact_value == "Uncertain":
                continue
            
            fact_nl = self.__retrieve_fact_from_rules(logic_expression=rule_expression, rule_nl=result['specific_rule'], predicate=f"{result[key]}({selected_name})", value=fact_value)
            
            distracting_facts.append(fact_nl)
            if fact_value == "True":
                distracting_facts_fol.append(f"{result[key]}({selected_name})")
            else:
                distracting_facts_fol.append(f"¬{result[key]}({selected_name})")
        
        # get result
        final_result = {'fact': distracting_facts, 'fact_fol': distracting_facts_fol}
        if self.__check_universal_rules(result['universal_rule']):
            final_result['rule'] = result['universal_rule']
            final_result['rule_fol'] = rule_expression
        else:
            final_result['rule'] = result['specific_rule']
            final_result['rule_fol'] = specific_rule_expression
        
        return final_result
        
    def __retrieve_fact_from_rules(self, logic_expression: str, rule_nl: str, predicate: str, value: str = None) -> str:
        messages = [
            {"role": "system", "content": "You will be provided a logic expression and its corresponding natural language. You need to translate the given query expression into natural language. Your answer should be in JSON format with keys: query_natural_language"}
        ]
        
        # uncertain facts can be either positive or negative
        if value == "uncertain":
            value = random.sample(['True', 'False'], 1)[0]
        
        if value is None or value == "True":  # RETRIEVE_FACT_EXAMPLES_POSITIVE
            few_shot_examples = [
                {"role": "user", "content": "logic expression: eat_apple(James) → (tummy_ache(James) ∨ sweet_taste(James))\nnatural language: If James eats an apple, then he will get a sweet taste or a tummy ache.\n\nquery expression: eat_apple(James)"},
                {'role': "assistant", "content": "{\n  \"query_natural_language\": \"James eats an apple.\"\n}"},
                {"role": "user", "content": "logic expression: lives_simply(Hunter) → (finds_contentment(Hunter) ⊕ is_respected(Hunter))\nnatural language: If Hunter lives simply, then he either finds contentment or is respected by his peers, but not both.\n\nquery expression: is_respected(Hunter)"},
                {'role': "assistant", "content": "{\n  \"query_natural_language\": \"Hunter is respected by his peers.\"\n}"},
                {"role": "user", "content": "logic expression: overcome_adversity(Hobson) ⊕ succumb_to_adversity(Hobson)\nnatural language: Hobson either overcomes adversity or succumbs to the endless adversity, but not both.\n\nquery expression: succumb_to_adversity(Hobson)"},
                {'role': "assistant", "content": "{\n  \"query_natural_language\": \"Hobson ultimately succumbs to the endless adversity.\"\n}"}
            ]
            query = {"role": 'user', 'content': f"logic expression: {logic_expression}\nnatural language: {rule_nl}\n\nquery expression: {predicate}"}
        else: # RETRIEVE_FACT_EXAMPLES_NEGATIVE 
            few_shot_examples = [
                {"role": "user", "content": "logic expression: eat_apple(James) → (tummy_ache(James) ∨ sweet_taste(James))\nnatural language: If James eats an apple, then he will get a sweet taste or a tummy ache.\n\nquery expression: ¬eat_apple(James)"},
                {'role': "assistant", "content": "{\n  \"query_natural_language\": \"James did not eat an apple.\"\n}"},
                {"role": "user", "content": "logic expression: lives_simply(Hunter) → (finds_contentment(Hunter) ⊕ is_respected(Hunter))\nnatural language: If Hunter lives simply, then he either finds contentment or is respected by his peers, but not both.\n\nquery expression: ¬is_respected(Hunter)"},
                {'role': "assistant", "content": "{\n  \"query_natural_language\": \"Hunter is not respected by his peers.\"\n}"},
                {"role": "user", "content": "logic expression: overcome_adversity(Hobson) ⊕ succumb_to_adversity(Hobson)\nnatural language: Hobson either overcomes adversity or succumbs to the endless adversity, but not both.\n\nquery expression: ¬succumb_to_adversity(Hobson)"},
                {'role': "assistant", "content": "{\n  \"query_natural_language\": \"Hobson does not succumb to the endless adversity.\"\n}"}
            ]
            query = {"role": 'user', 'content': f"logic expression: {logic_expression}\nnatural language: {rule_nl}\n\nquery expression: ¬{predicate}"}
            
        messages.extend(few_shot_examples)
        messages.append(query)
        
        result = self.__send_request(messages)
        return result['query_natural_language']
    
    def __check_universal_rules(self, rule_nl: str) -> bool:
        prompt = [
            {"role": "system", "content": "You will be given a natural language sentence. Your task is to decide whether the rule expressed in the given sentence is a widely accepted common sense.\n\nYour answer should be in JSON format with keys: answer"}
        ]
        
        prompt.extend(self.examples['rule_checking'])
        prompt.append({"role": "user", "content": rule_nl})
        
        result = self.__send_request(prompt)
        
        if result['answer'] == 'True':
            return True
        else:
            return False
    
    def __send_request(self, message: List) -> str:
        while True:
            api_flag = False
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    temperature=0.7,
                )
                answer_str = completion.choices[0].message.content.replace("```json", "").replace("```", "")
                result = eval(answer_str)
                api_flag = True
            except:
                self.err_cnt += 1
                if api_flag == False:
                    print(f"API error occured, wait for 2 seconds. Error count: {self.err_cnt}")
                time.sleep(2)
                
            if api_flag:
                break
            
        return result
    
    @staticmethod
    def extract_facts(rule_expression: str) -> List:
        pattern = r'\[F\d+\]'
        matches = re.findall(pattern, rule_expression)
        result = [match.strip('[]') for match in matches]
        
        pattern = r'\[D\d+\]'
        matches = re.findall(pattern, rule_expression)
        result.extend([match.strip('[]') for match in matches])
        
        def get_position(element):
            return rule_expression.find(f'[{element}]')
        
        return sorted(result, key=get_position)
    
    @staticmethod
    def load_json(filepath: str) -> List:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        return data