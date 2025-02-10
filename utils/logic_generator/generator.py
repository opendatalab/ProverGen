import re
import random
from typing import List, Dict, Tuple

import numpy as np
from tqdm.auto import tqdm

from utils.my_dataclass import Problems
from .rule_generator import RuleGenerator


class LogicGenerator:
    
    def __init__(self, args) -> None:
        self.args = args
        # problem property
        self.reasoning_depth = {
            'easy': [1, 2],
            'medium': [3, 4, 5],
            'hard': [6, 7, 8, 9]
        }
        
        # goal value property
        self.goal_value_dict = {0: 'True', 1: 'False', 2: 'Uncertain'}
        self.goal_value_probs = args.goal_value_probs
        
        # rule generator
        self.rule_generator = RuleGenerator(args)
        
    def generate_logic_skeletons(self, verbose: bool = False) -> List[Problems]:
        """ The generation of problems involve three steps:
        1. Decide the property of the goal, i.e. the truth value and the type (e.g., whether the goal is a fact or a rule).
        2. Generate the first rule according the properties of the goal.
        3. Iteratively generate rules until reaching the maximum reasoning steps.
        """
        problems = []
        problem_generation_bar = tqdm(range(self.args.num))
        
        while len(problems) < self.args.num:
            # 1 problem property
            rule_id = 0
            fact_num = 0
            problem_rules = []
            # goal property
            goal_value = self.sample_goal_value()
            rule_as_goal = np.random.choice(
                a=np.array([0, 1]),
                size=1,
                replace=True,
                p=self.args.rule_as_goal_proportion
            )
            
            # 2 generate the first rule according to the type of the final goal
            if rule_as_goal == 0:
                rule_expression, rule_requirement, goal_index, fact_num = self.rule_generator.generate_normal_rule(goal_expression="", gvalue=goal_value, fact_num=fact_num)
                
                if rule_expression is None:  # the condition cannot be satisfied, skip this one and resample
                    continue
                
                root_rule = {'id': rule_id, 'next_rule': None, 'expression': rule_expression, 'value': self.get_fact_value(rule_expression, rule_requirement, goal_index), 'conclusion': {'index': goal_index, 'expression': self.extract_facts(rule_expression)[goal_index], 'value': goal_value}}
                
                # get one reasoning step and final goal
                problem_answers = [self.get_single_deduction_step(rule_expression=rule_expression, rule_requirement=rule_requirement, conclusion_value=root_rule['conclusion']['value'], goal_index=goal_index)]
                step_cnt = 1
            else:
                rule_expression, rule_requirement, fact_num = self.rule_generator.generate_goal_rule(gvalue=goal_value, fact_num=fact_num)
                root_rule = {'id': rule_id, 'next_rule': None, 'expression': rule_expression, 'value': self.get_fact_value(rule_expression, rule_requirement), 'conclusion': {'index': None, 'expression': rule_expression, 'value': goal_value}}
                
                # get reasoning step and final goal
                problem_answers = [self.get_single_deduction_step(rule_expression=rule_expression, rule_requirement=rule_requirement, conclusion_value=root_rule['conclusion']['value'])]
                step_cnt = 0
                
            # get the goal of the problem
            final_goal = root_rule['conclusion']
                
            problem_rules.append(root_rule)
            root_rule_facts = self.extract_facts(rule_expression)
            facts_pool = [{'rule': rule_id, 'fact_position': i, 'expression': root_rule_facts[i], 'value': root_rule['value'][i]} for i in root_rule['value'].keys()]
            rule_id += 1
            
            # 3 Iteratively generate remaining rules
            steps = random.sample(self.reasoning_depth[self.args.mode], 1)[0]
            final_facts = []
            dead_end = False
            
            while step_cnt < steps:
                # Randomly select a fact for proving
                goal = facts_pool.pop(random.sample(list(range(len(facts_pool))), 1)[0])
                
                if (len(facts_pool) >= self.args.fact_num_threshold) and (np.random.uniform() < self.args.fact_num_prob):
                    # The current fact will be given directly
                    final_facts.append(goal)
                    continue
                else:
                    rule_expression, rule_requirement, goal_index, fact_num = self.rule_generator.generate_normal_rule(goal_expression=goal['expression'], gvalue=goal['value'], fact_num=fact_num)
                    
                    if rule_expression is None:
                        dead_end = True
                        break
                    
                    # Generate distracting rule
                    distracting_rule_expression, distracting_rule_requirement, distracting_goal_index = self.rule_generator.generate_distracting_rules(goal_expression=goal['expression'])
                    distracting_rule = {'expression': distracting_rule_expression, 'value': self.get_fact_value(distracting_rule_expression, distracting_rule_requirement, distracting_goal_index)}
                    
                    # create current_rule
                    current_rule_facts = self.extract_facts(rule_expression)
                    current_rule = {'id': rule_id, 'next_rule': goal['rule'], 'expression': rule_expression, 'value': self.get_fact_value(rule_expression, rule_requirement, goal_index), 'conclusion': {'index': goal_index, 'expression': current_rule_facts[goal_index], 'value': goal['value']}, 'distracting_rule': distracting_rule}
                    
                    problem_rules.append(current_rule)
                    facts_pool.extend([{'rule': rule_id, 'fact_position': i, 'expression': current_rule_facts[i], 'value': current_rule['value'][i]} for i in current_rule['value'].keys()])
                    rule_id += 1

                    # get problem answer
                    problem_answers.insert(0, self.get_single_deduction_step(rule_expression=rule_expression, rule_requirement=rule_requirement, conclusion_value=current_rule['conclusion']['value'], goal_index=goal_index))

                    step_cnt += 1
            
            if dead_end:
                continue
            # save problem
            final_facts.extend(facts_pool)
            problems.append(Problems(
                id=len(problems),
                goal=final_goal,
                facts=final_facts,
                rules=problem_rules,
                reasoning_chain=problem_answers,
            ))
            
            # print current problems
            if verbose:
                print("Facts:")
                for item in problems[-1].facts:
                    print(f"{item['expression']}: {item['value']}")
                    
                print("\nRules:")
                for item in problems[-1].rules:
                    print(f"{item['expression']} |-> {item['conclusion']['expression']} | {item['conclusion']['value']}")
                    
                print("\nConclusion:")
                print(f"{problems[-1].goal['expression']}: {problems[-1].goal['value']}")
                
                print("\nReasoning Chain:")
                print("==========================")
                for item in problems[-1].reasoning_chain:
                    print("fact: ", end="")
                    for fact in item['facts']:
                        print(f"{fact['expression']}|{fact['value']}", end=' ')
                    print(f"\nrule: {item['rule']}")
                    print(f"conclusion: {item['conclusion']['expression']} | {item['conclusion']['value']}")
                    print("==========================")
            
            problem_generation_bar.update(1)
            
        return problems
            
    def sample_goal_value(self) -> str:
        # Sample goal value from [True, False, Uncertain] according to the given distribution
        goal_value = np.random.choice(
            a=np.array([0, 1, 2]),
            size=1,
            replace=True,
            p=self.goal_value_probs
        )
        
        return self.goal_value_dict[goal_value[0]]
    
    def get_single_deduction_step(self, rule_expression: str, rule_requirement: list, conclusion_value: str, goal_index: int = None) -> Dict:
        """
        Get the deduction step according to the rule and the fact.
        """
        current_facts_list = self.extract_facts(rule_expression)
        rule_fact = []
                    
        for i in range(len(current_facts_list)):
            if i == goal_index:
                continue
            else:
                rule_fact.append({'expression': current_facts_list[i], 'value': rule_requirement[len(rule_fact)]})
        
        single_step_deduction = {'facts': rule_fact, 'rule': rule_expression, 'conclusion': {'expression': rule_expression if goal_index is None else current_facts_list[goal_index], 'value': conclusion_value}}
        
        return single_step_deduction
    
    @staticmethod
    def extract_facts(rule_expression: str) -> List:
        """
        Extract all the facts in the rule
        """
        pattern = r'\[F\d+\]'
        matches = re.findall(pattern, rule_expression)
        result = [match.strip('[]') for match in matches]
        
        return result
        
    @staticmethod
    def get_fact_value(rule: str, value_tuple: Tuple, gindex: int = None) -> Dict:
        """
        Assign the values to the facts in the rule.
        """
        result = {}
        if gindex is None:  # The rule is the final conclusion
            for i in range(rule.count('[F')):
                result[i] = value_tuple[i]
        else:
            value_list = []
            for item in value_tuple:
                value_list.append(item)
            value_list.insert(gindex, None)
            
            for i in range(rule.count('[F')):
                if i == gindex:
                    continue
                else:
                    result[i] = value_list[i]
        return result