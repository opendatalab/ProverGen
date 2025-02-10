import os
import re
import json
import time
import random
from typing import Dict, List, Tuple

from tqdm.auto import tqdm
from openai import OpenAI


class Translator:
    
    def __init__(self, args) -> None:
        self.args = args
        self.model_name = args.model_name
        self.names = self.__load_names(args.name_path)
        self.predicates = self.__load_predicates(args.predicate_path)
        self.examples = self.load_examples(args.example_path)
        
        if args.api_key == "EMPTY" and args.base_url == "EMPTY":
            self.client = OpenAI()
        else:
            self.client = OpenAI(
                api_key=args.api_key,
                base_url=args.base_url
            )
        self.err_cnt = 0
        
    def translate_rules_and_facts(self, data) -> List:
        """
        1. Randomly select a name and generate background story
        2. Translate root rule and facts
        3. Translate other rules and facts
        4. Get facts in the context
        """
        translated_problems = []
        
        problem_bar = tqdm(range(len(data)))
        for i in problem_bar:
            if i < self.args.start or i >= self.args.end:
                continue
            problem = data[i]
            
            # 1 background story
            selected_name = random.sample(self.names, 1)[0]
            name_category, background_story, background_keyword = self.__generate_background(selected_name)
            
            translated_rules = []
            translated_facts = {}
            rule_expression_to_id = {}
            
            # 2: root rule
            """translate the root rule"""
            root_rule = problem.rules[0]
            root_rule_facts = self.extract_facts(root_rule['expression'])
            rule_expression_to_id[root_rule['expression']] = root_rule['id']  # map rule expression to id
            
            root_rule, rule_facts, conclusion, problem_answer = self.__translate_root_rule(root_rule, selected_name, background_story=background_story, name_category=name_category)
            translated_rules.append(root_rule)
            
            # save rule fact
            for key in rule_facts.keys():
                if key != root_rule['conclusion']['expression']:
                    fact_nl, fact_value = self.__retrieve_fact_from_rules(logic_expression=root_rule['specific_expression'], rule_nl=root_rule['specific_nl'], predicate=f"{rule_facts[key]}({selected_name})", value=root_rule['value'][root_rule_facts.index(key)])

                    if fact_value == "False":
                        translated_facts[key] = {'expression': rule_facts[key], 'rule': root_rule['specific_expression'], 'rule_nl': root_rule['specific_nl'], 'fact': f"¬{rule_facts[key]}({selected_name})", 'fact_nl': fact_nl}
                    else:
                        translated_facts[key] = {'expression': rule_facts[key], 'rule': root_rule['specific_expression'], 'rule_nl': root_rule['specific_nl'], 'fact': f"{rule_facts[key]}({selected_name})", 'fact_nl': fact_nl}
                else:
                    translated_facts[key] = {'expression': rule_facts[root_rule['conclusion']['expression']], 'rule': root_rule['specific_expression'], 'rule_nl': root_rule['specific_nl'], 'fact': conclusion['expression'], 'fact_nl': conclusion['nl']}
            
            # 3: other rule and fact
            """ translate other rules """
            for rule_id in range(1, len(problem.rules)):
                current_rule = problem.rules[rule_id]
                rule_expression_to_id[current_rule['expression']] = current_rule['id']
            
                current_rule, translated_facts = self.__translate_normal_rule(current_rule, translated_facts, selected_name=selected_name, background_story=background_story, name_category=name_category)
                translated_rules.append(current_rule)
            
            # 4: Get facts in the context
            """ get provided facts """
            context_facts = []
            context_facts_fol = []
            for idx in range(len(problem.facts)):
                if problem.facts[idx]['value'] != "Uncertain":
                    fact_idx = problem.facts[idx]['expression']
                    context_facts.append(translated_facts[fact_idx]['fact_nl'])
                    context_facts_fol.append(translated_facts[fact_idx]['fact'])
                    
            # save result
            translated_problems.append(
                {
                    "name": selected_name,
                    "keyword": background_keyword,
                    "background": background_story,
                    "name_category": name_category,
                    "context_facts": context_facts,
                    "context_facts_fol": context_facts_fol,
                    "translated_facts": translated_facts,
                    "translated_rules": translated_rules,
                    "reasoning_chain": problem.reasoning_chain,
                    "rule_expression_to_id": rule_expression_to_id,
                    "conclusion": conclusion,
                    "answer": problem_answer
                }
            )
        
        return translated_problems
        
    def __translate_root_rule(self, rule: Dict, name: str, background_story: str, name_category: str) -> Tuple:
        """
        1. Translate rule
        2. Extract facts, i.e. predicates
        3. Get specific rule
        """
        rule_expression = rule['expression']
        rule_facts = self.extract_facts(rule_expression)
        
        # transform rule
        for item in rule_facts:
            rule_expression = rule_expression.replace(f"[{item}]", f"{item}(x)")
        rule_expression = f"∀x ({rule_expression})"
        
        # get few shot examples
        few_shot_examples = self.examples['root'][rule_expression]
        
        translate_logic_prompt = [
            {"role": "system", "content": f"You will be provided a background story and a logic expression, your task is to replace the placeholders in the logic expression with appropriate words so that this logic expression represents a real world common sense rule and does not conflict with the common sense knowledge. The selected words should have no more than 5 words. If the chosen words can have some connection to the background story, that would be best, but it's okay if they don't. Note: Your answer should be in JSON format. The keys should be the placeholders (i.e. {rule_facts}) and the natural language of the rule."}
        ]
        translate_logic_prompt.extend(few_shot_examples)
        translate_logic_prompt.append({"role": "user", "content": f"background story:\n{background_story}\n\nlogic expression: {rule_expression}\nNote: x belongs to {name_category}."})
        
        result = self.__send_request(message=translate_logic_prompt)
        
        # 2: extract facts
        fact_list = {}
        for key in result.keys():
            if key in ['universal_rule', 'specific_rule']:
                continue
            result[key] = result[key].replace(" ", "_").replace(".", "_").replace("'", "").replace("-", "_")
            rule_expression = rule_expression.replace(key, result[key])  # replace the placeholders with translated predicates
            fact_list[key] = result[key]
        
        # 3: get specific rule
        specific_rule_expression = rule_expression.replace(f"∀x (", "")[:-1]
        specific_rule_expression = specific_rule_expression.replace("(x)", f"({name})")
        # get result
        rule['universal_expression'] = rule_expression
        rule['universal_nl'] = result['universal_rule']
        rule['specific_expression'] = specific_rule_expression
        rule['specific_nl'] = result['specific_rule']
        
        # 4?
        if rule['conclusion']['index'] is None:  # the rule itself is the goal
            rule['conclusion']['nl'] = result['specific_rule']
            conclusion = {'expression': specific_rule_expression, 'nl': result['specific_rule'], 'value': rule['conclusion']['value']}
            
            answer = rule['conclusion']['value']
        else:
            selected_value = random.sample(["True", "False"], 1)[0]  # add negation or not
            
            fact_nl, fact_value = self.__retrieve_fact_from_rules(logic_expression=specific_rule_expression, rule_nl=result['specific_rule'], predicate=f"{result[rule['conclusion']['expression']]}({name})", value=selected_value)
            
            if selected_value == "True":
                conclusion = {'expression': f"{result[rule['conclusion']['expression']]}({name})", 'nl': fact_nl, 'value': rule['conclusion']['value']}
            else:
                conclusion = {'expression': f"¬{result[rule['conclusion']['expression']]}({name})", 'nl': fact_nl, 'value': rule['conclusion']['value']}
            
            if rule['conclusion']['value'] == "Uncertain":
                answer = "Uncertain"
            elif selected_value == rule['conclusion']['value']:
                answer = "True"
            else:
                answer = "False"
        
        return rule, fact_list, conclusion, answer
    
    def __translate_normal_rule(self, rule: Dict, translated_facts: Dict, selected_name: str, background_story: str, name_category: str) -> Tuple:
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
        
        # 1: get forbiden list
        forbidden_list = []
        for key in translated_facts.keys():
            forbidden_list.append(translated_facts[key]['expression'])
        
        # 2: transform rules and get references
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
        
        # To maintain predicate consistency across rules, it is imperative to reference existing translations, as some facts may have been previously translated.
        assert len(existed_predicate) == 1
        references = ""
        for item in existed_predicate:
            references += f"{translated_facts[item]['expression']}\n{translated_facts[item]['rule']}: {translated_facts[item]['rule_nl']}\n"
        references = references[:-1]
        
        # 3: Translate rules
        # get few shot examples
        example_index = ""
        for chr in rule_expression:
            if chr in ["→", "⊕", "∨", "∧"]:
                example_index += chr
        
        few_shot_examples = self.examples['normal'][str(example_index)]
        
        # construct prompt
        translate_logic_prompt = [
            {"role": "system", "content": "You will be provided a logic expression, a reference of the existed predicate in the logic expression, and a background story. Your task is to replace the placeholders in the logic expression with appropriate predicates (no more than 5 words) so that the provided logic expression represent a real world common sense rule.\n\nDo not use 'not'. Do not use the words in the forbidden list. The words that are similar in meaning to the words in the forbidden list are also not allowed.\n\nYour answer is not required to closely connected to the background story. You can use any predicates you like as long as their length is less than 5.\n\nYour answer should be in JSON format with the provided keys."}
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
        
        # 4: Get the translation result
        # get rule expression and new fact
        for key in result.keys():
            if key in ["universal_rule", "specific_rule"]:
                continue
            result[key] = result[key].replace(" ", "_").replace(".", "_").replace("'", "").replace("-", "_")
            rule_expression = rule_expression.replace(key, result[key])
            
        specific_rule_expression = rule_expression.replace(f"∀x (", "")[:-1]
        specific_rule_expression = specific_rule_expression.replace("(x)", f"({selected_name})")
        
        for key in result.keys():
            if key in ["universal_rule", "specific_rule"]:
                continue
            
            fact_nl, fact_value = self.__retrieve_fact_from_rules(logic_expression=rule_expression, rule_nl=result['specific_rule'], predicate=f"{result[key]}({selected_name})", value=rule['value'][rule_facts.index(key)])
            
            if fact_value == "True":
                translated_facts[key] = {'expression': result[key], 'rule': specific_rule_expression, 'rule_nl': result['specific_rule'], 'fact': f"{result[key]}({selected_name})", "fact_nl": fact_nl}
            else:
                translated_facts[key] = {'expression': result[key], 'rule': specific_rule_expression, 'rule_nl': result['specific_rule'], 'fact': f"¬{result[key]}({selected_name})", "fact_nl": fact_nl}
        
        # get result
        rule['universal_expression'] = rule_expression
        rule['universal_nl'] = result['universal_rule']
        rule['specific_expression'] = specific_rule_expression
        rule['specific_nl'] = result['specific_rule']
        
        return rule, translated_facts    
        
    def __retrieve_fact_from_rules(self, logic_expression: str, rule_nl: str, predicate: str, value: str) -> Tuple:
        """
        Get the natural language of predicates according to the rule and corresponding natural language
        """
        messages = [
            {"role": "system", "content": f"You will be provided a logic expression and its corresponding natural language. You need to translate the given query expression into natural language. The predicates in the logic expression should appear in the natural language sentence. Your answer should be in JSON format with keys: query_natural_language"}
        ]
        
        # uncertain facts can be either positive or negative
        if value == "Uncertain":
            value = random.sample(['True', 'False'], 1)[0]
        
        if value == "True":
            few_shot_examples = [
                {"role": "user", "content": f"logic expression: eat_apple(James) → (tummy_ache(James) ∨ sweet_taste(James))\nnatural language: If James eats an apple, then he will get a sweet taste or a tummy ache.\n\nquery expression: eat_apple(James)"},
                {'role': "assistant", "content": "{\n  \"query_natural_language\": \"James eats an apple.\"\n}"},
                {"role": "user", "content": f"logic expression: lives_simply(Hunter) → (finds_contentment(Hunter) ⊕ is_respected(Hunter))\nnatural language: If Hunter lives simply, then he either finds contentment or is respected by his peers, but not both.\n\nquery expression: is_respected(Hunter)"},
                {'role': "assistant", "content": "{\n  \"query_natural_language\": \"Hunter is respected by his peers.\"\n}"},
                {"role": "user", "content": f"logic expression: overcome_adversity(Hobson) ⊕ succumb_to_adversity(Hobson)\nnatural language: Hobson either overcomes adversity or succumbs to the endless adversity, but not both.\n\nquery expression: succumb_to_adversity(Hobson)"},
                {'role': "assistant", "content": "{\n  \"query_natural_language\": \"Hobson ultimately succumbs to the endless adversity.\"\n}"}
            ]
            query = {"role": 'user', 'content': f"logic expression: {logic_expression}\nnatural language: {rule_nl}\n\nquery expression: {predicate}"}
        else:
            few_shot_examples = [
                {"role": "user", "content": f"logic expression: eat_apple(James) → (tummy_ache(James) ∨ sweet_taste(James))\nnatural language: If James eats an apple, then he will get a sweet taste or a tummy ache.\n\nquery expression: ¬eat_apple(James)"},
                {'role': "assistant", "content": "{\n  \"query_natural_language\": \"James did not eat an apple.\"\n}"},
                {"role": "user", "content": f"logic expression: lives_simply(Hunter) → (finds_contentment(Hunter) ⊕ is_respected(Hunter))\nnatural language: If Hunter lives simply, then he either finds contentment or is respected by his peers, but not both.\n\nquery expression: ¬is_respected(Hunter)"},
                {'role': "assistant", "content": "{\n  \"query_natural_language\": \"Hunter is not respected by his peers.\"\n}"},
                {"role": "user", "content": f"logic expression: overcome_adversity(Hobson) ⊕ succumb_to_adversity(Hobson)\nnatural language: Hobson either overcomes adversity or succumbs to the endless adversity, but not both.\n\nquery expression: ¬succumb_to_adversity(Hobson)"},
                {'role': "assistant", "content": "{\n  \"query_natural_language\": \"Hobson does not succumb to the endless adversity.\"\n}"}
            ]
            query = {"role": 'user', 'content': f"logic expression: {logic_expression}\nnatural language: {rule_nl}\n\nquery expression: ¬{predicate}"}
            
        messages.extend(few_shot_examples)
        messages.append(query)
        
        result = self.__send_request(messages)
        return result['query_natural_language'], value
    
    def __generate_background(self, name: str) -> Tuple:
        # Generate a background story for the given name
        keyword = random.sample(self.predicates, 1)[0]

        prompt = [
            {"role": "system", "content": "You will be given a keyword and a name (can be a person's name or an animals' name). Your tasks is to generate a background story with no more than 150 words according to the keyword about this entity. Your answer should be in JSON format with keys: category, story."}
        ]
        prompt.extend(self.examples['story'])
        prompt.append({"role": 'user', "content": f"keyword: {keyword}\nname: {name}"})
        
        result = self.__send_request(prompt)
        return result['category'], result['story'], keyword
    
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
                    print(f"API error occured, wait for 3 seconds. Error count: {self.err_cnt}")
                time.sleep(2)
                
            if api_flag:
                break
            
        return result
    
    def __load_predicates(self, filepath: str) -> List:
        return self.load_json_file(filepath)['words']
    
    def __load_names(self, file_dir: str) -> List:
        name_list = []
        filenames = os.listdir(file_dir)

        for filename in filenames:
            filepath = os.path.join(file_dir, filename)
            c_name_list = self.load_json_file(filepath)['names']
            for item in c_name_list:
                if item not in name_list:
                    name_list.append(item)

        def contains_special_characters(s):
            match = re.search(r'[^a-zA-Z]', s)
            return match is not None
        
        filtered_name_list = []
        for item in name_list:
            if not contains_special_characters(item):
                filtered_name_list.append(item)
            
        return filtered_name_list
    
    def load_examples(self, filepath: str) -> Dict:
        return self.load_json_file(filepath)
    
    @staticmethod
    def extract_facts(rule_expression: str) -> List:
        pattern = r'\[F\d+\]'
        matches = re.findall(pattern, rule_expression)
        result = [match.strip('[]') for match in matches]
        
        return result
    
    @staticmethod
    def load_json_file(file_path: str) -> List:
        with open(file_path, 'r') as f:
            word_list = json.load(f)
        
        return word_list

