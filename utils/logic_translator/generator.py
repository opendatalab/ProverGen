import json
import time
from typing import List

from tqdm.auto import tqdm
from openai import OpenAI


class ProblemGenerator:
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
    
    def create_problems(self) -> List:
        """
        1. Select facts
        2. Select rules, including selecting universal or specific rules
        3. Select distractions
        4. Get reasoning chains
        5. Get conclusions
        6. Generate Context
        """
        result = []
        
        pbar = tqdm(range(len(self.data)))
        for item in self.data:
            translated_facts = item['translated_facts']
            translated_rules = item['translated_rules']
            rule_expression_to_id = item['rule_expression_to_id']
            
            current_problem = {}
            
            # 1: get facts
            current_problem['facts'] = item['context_facts']
            current_problem['facts_fol'] = item['context_facts_fol']
            
            # 2: get rules
            rules = []
            rules_fol = []
            selected_rule = {}
            for rule in translated_rules:
                if rule['conclusion']['expression'].count("[F") >= 2:
                    continue
                if self.__check_universal_rules(rule['universal_nl']):
                    rules.append(rule['universal_nl'])
                    rules_fol.append(rule['universal_expression'])
                    selected_rule[rule['id']] = "universal"
                else:
                    rules.append(rule['specific_nl'])
                    rules_fol.append(rule['specific_expression'])
                    selected_rule[rule['id']] = "specific"
                    
            current_problem['rules'] = rules
            current_problem['rules_fol'] = rules_fol
            
            # 3: get distracting facts and rules
            distracting_facts = []
            distracting_facts_fol = []
            distracting_rules = []
            distracting_rules_fol = []
            for rule in translated_rules:
                if 'distracting_rule' in rule.keys():
                    d_rule = rule['distracting_rule']
                    for d_i in range(len(d_rule['fact'])):
                        distracting_facts.append(d_rule['fact'][d_i])
                        distracting_facts_fol.append(d_rule['fact_fol'][d_i])

                    distracting_rules.append(d_rule['rule'])
                    distracting_rules_fol.append(d_rule['rule_fol'])
            
            current_problem['distracting_facts'] = distracting_facts
            current_problem['distracting_facts_fol'] = distracting_facts_fol
            current_problem['distracting_rules'] = distracting_rules
            current_problem['distracting_rules_fol'] = distracting_rules_fol
            
            # 4: get reasoning chains
            reasoning_chains = []
            reasoning_chains_fol = []
            for reasoning_step in item['reasoning_chain']:
                # get step fact
                step_facts = []
                step_facts_fol = []
                for fact in reasoning_step['facts']:
                    if fact['value'] == "Uncertain":
                        continue
                    step_facts.append(translated_facts[fact['expression']]['fact_nl'])
                    step_facts_fol.append(translated_facts[fact['expression']]['fact'])
                
                # get step rule
                sc_expression = reasoning_step['conclusion']['expression']
                if sc_expression.count("[F") >= 2:
                    step_conclusion_rule = translated_rules[rule_expression_to_id[sc_expression]]
                    step_conclusion = step_conclusion_rule['specific_nl']
                    step_conclusion_fol = step_conclusion_rule['specific_expression']
                    
                    if reasoning_step['conclusion']['value'] == "Uncertain":
                        step_conclusion = None
                        step_conclusion_fol = None
                        
                    reasoning_chains.append({"facts": step_facts, "rules": None, "conclusion": step_conclusion})
                    reasoning_chains_fol.append({"facts": step_facts_fol, "rules": None, "conclusion": step_conclusion_fol})
                else:
                    rule_id = rule_expression_to_id[reasoning_step['rule']]
                    step_rule = translated_rules[rule_id][f"{selected_rule[rule_id]}_nl"]  # check
                    step_rule_fol = translated_rules[rule_id][f"{selected_rule[rule_id]}_expression"]
                
                    step_conclusion = translated_facts[sc_expression]['fact_nl']
                    step_conclusion_fol = translated_facts[sc_expression]['fact']
                    
                    if reasoning_step['conclusion']['value'] == "Uncertain":
                        step_conclusion = None
                        step_conclusion_fol = None
                        
                    reasoning_chains.append({"facts": step_facts, "rules": step_rule, "conclusion": step_conclusion})
                    reasoning_chains_fol.append({"facts": step_facts_fol, "rules": step_rule_fol, "conclusion": step_conclusion_fol})
            
            current_problem['reasoning_chains'] = reasoning_chains
            current_problem['reasoning_chains_fol'] = reasoning_chains_fol
            
            # 5: Get conclusions
            current_problem["conclusion"] = item['conclusion']['nl']
            current_problem["conclusion_fol"] = item['conclusion']['expression']
            current_problem["answer"] = item['answer']
            
            # 6: Generate Context
            context = []
            context_fol = []
            for i in range(len(reasoning_chains)):
                ordered_fact = []
                ordered_fact_fol = []
                for j in range(len(reasoning_chains_fol[i]['facts'])):
                    if reasoning_chains_fol[i]['facts'][j] in current_problem['facts_fol']:
                        ordered_fact.append(reasoning_chains[i]['facts'][j])
                        ordered_fact_fol.append(reasoning_chains_fol[i]['facts'][j])
                context.extend(ordered_fact)
                context_fol.extend(ordered_fact_fol)
                
                ordered_rules = reasoning_chains[i]['rules']
                if ordered_rules in current_problem['rules']:
                    context.append(ordered_rules)
                    context_fol.append(reasoning_chains_fol[i]['rules'])
            
            assert len(context) == len(context_fol)
            assert len(context) == len(current_problem['facts']) + len(current_problem['rules'])
            
            for fact_item in current_problem['facts']:
                assert fact_item in context
            for rule_item in current_problem['rules']:
                assert rule_item in context
            
            current_problem['context'] = context
            current_problem['context_fol'] = context_fol
            
            # background story and corresponding name
            current_problem['background_story'] = item['background']
            current_problem['name'] = item['name']
            current_problem['keyword'] = item['keyword']
            current_problem['subject_category'] = item['name_category']
            
            result.append(current_problem)
            pbar.update(1)
            
        return result
    
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
    def load_json(filepath: str) -> List:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        return data