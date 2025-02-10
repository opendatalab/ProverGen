import json
import random
from typing import Tuple, List, Dict
from itertools import product

from utils.prover9_interface import prover9_prove, FOL2Prover9Converter


class RuleGenerator:
    
    def __init__(self, args) -> None:
        self.args = args
        self.converter = FOL2Prover9Converter()
        self.fact_holder = ["aget(x)", "bget(x)", "cget(x)"]
        self.load_rule_candidate(args.rule_candidate_path)
        
        
    def generate_goal_rule(self, gvalue: str, fact_num: int) -> Tuple:
        """
        1. Randomly sample a rule from the pool
        2. Calculate the truth value of the sampled rule
        3. Replace placeholders [F] with [F1], [F2], ..., [Fn]
        4. Randomly sample 1 solution as the final result
        """
        solutions = []
        err_cnt = -1
        
        while solutions == []:
            err_cnt += 1
            if err_cnt >= 50:
                return None, None, None
            logic_expression = random.sample(self.goal_rules, 1)[0]
            solutions = self.__calculate_root_rule_truth_table(logic_expression, gvalue)
        
        expression_fact_num = logic_expression.count('[F]')
        fact_num_new = fact_num + expression_fact_num
        for i in range(fact_num, fact_num_new):
            logic_expression = logic_expression.replace('[F]', f'[F{fact_num + i}]', 1)
        
        value_requirements = random.sample(solutions, 1)[0]
        return logic_expression, value_requirements, fact_num_new
    
    def generate_normal_rule(self, goal_expression: str, gvalue: str, fact_num: int) -> Tuple:
        """ 
        1. Randomly sample a rule from the pool and then decide the position of the conclusion
        2. Calculate the solutions for the given gvalue and sampled expression
        3. Replace placeholders [F] with [F1], [F2], ..., [Fn]
        4. Randomly sample 1 solution as the final result
        """
        err_cnt = -1
        solutions = []
        while solutions == []:
            err_cnt += 1
            if err_cnt >= 50:
                return None, None, None, None

            logic_expression = random.sample(self.normal_rules, 1)[0]
            if self.args.mode == "hard":
                f_cnt = logic_expression.count("[F]")
                g_possible_position = list(range(1, f_cnt + 1))
                
                goal_index = random.sample([-gpp for gpp in g_possible_position], 1)[0]
            else:
                if logic_expression in ["[F] → ([F] ∧ [F])", "[F] → ([F] ∨ [F])", "[F] → ([F] ⊕ [F])"]:
                    goal_index = random.sample([-1, -2], 1)[0]
                else:
                    goal_index = -1
                    
            goal_index += logic_expression.count('[F]')
            solutions = self.__calculate_normal_rule_truth_table(logic_expression, gindex=goal_index, gvalue=gvalue)
        
        expression_fact_num = logic_expression.count('[F]')
        fact_num_new = fact_num + expression_fact_num if goal_expression == "" else fact_num + expression_fact_num - 1
        
        # Introduce new facts in the expression, we use F1, F2, ... to represent different facts
        offset = 0  # control the number of the newly introduced fact
        for i in range(fact_num, fact_num + expression_fact_num):
            if i - fact_num == goal_index and goal_expression != "":
                logic_expression = logic_expression.replace('[F]', f"[{goal_expression}]", 1)
                offset += 1
            else:
                logic_expression = logic_expression.replace('[F]', f'[F{i - offset}]', 1)
        
        # Randomly sample 1 solution as the final result
        value_requirements = random.sample(solutions, 1)[0]
        return logic_expression, value_requirements, goal_index, fact_num_new
    
    def generate_distracting_rules(self, goal_expression: str) -> Tuple:
        """
        1. Randomly sample a rule from the pool
        2. Decide the position of the goal
        3. Calculate the truth value of the rule
        4. Replace [F] with [F1], [F2], ..., [Fn]
        5. Randomly sample 1 solution as the final result
        """
        err_cnt = -1
        solutions = []
        while solutions == []:
            err_cnt += 1
            if err_cnt >= 50:
                return None, None, None, None

            logic_expression = random.sample(self.normal_rules, 1)[0]
            if logic_expression in ["[F] → ([F] ∧ [F])", "[F] → ([F] ∨ [F])", "[F] → ([F] ⊕ [F])"]:
                goal_index = random.sample([-1, -2], 1)[0]
            else:
                goal_index = -1
            goal_index += logic_expression.count('[F]')
                
            solutions = self.__calculate_normal_rule_truth_table(logic_expression, gindex=goal_index, gvalue="Uncertain")
        
        expression_fact_num = logic_expression.count('[F]')
        
        offset = 0  # control the ID of the fact
        for i in range(0, expression_fact_num):
            if i == goal_index and goal_expression != "":
                logic_expression = logic_expression.replace('[F]', f"[{goal_expression}]", 1)
            else:
                logic_expression = logic_expression.replace('[F]', f'[D{offset}]', 1)
                offset += 1
        
        value_requirements = random.sample(solutions, 1)[0]
        return logic_expression, value_requirements, goal_index
        
    def __calculate_normal_rule_truth_table(self, rule: str, gindex: int, gvalue: str) -> List:
        fact_num = rule.count('[F]')
        # replace [F] with placeholder fact
        for i in range(fact_num):
            rule = rule.replace("[F]", self.fact_holder[i], 1)
        rule = f"∀x ({rule})"
        rule = self.converter.convert_expression(rule)
        
        # get truth table with the help of theorem prover
        value_table = list(product(['True', 'False', 'Uncertain'], repeat=fact_num - 1))
        truth_table = []
        
        for item in value_table:
            placeholder_facts = self.fact_holder[:fact_num]
            goal_fact = placeholder_facts.pop(gindex)
            
            premises = self.assign_rule_value(item, fact_list=placeholder_facts)
            premises.insert(0, rule)
            arguments = [((f"all x. {goal_fact}"), premises)]
            
            prover9_result = self.prover9(arguments_list=arguments)
            
            # goal value is uncertain means that even if we know the rule and the remaining fact, we still cannot deduce the target's value.
            if gvalue == "Uncertain":
                rule = premises.pop(0)
                arguments = [((rule), premises)]
                check_result = self.prover9(arguments_list=arguments)
                if check_result == 'True':
                    truth_table.append(
                        {
                            'value': item,
                            'result': prover9_result
                        }
                    )
            else:
                truth_table.append(
                    {
                        'value': item,
                        'result': prover9_result
                    }
                )
        
        result = []
        for item in truth_table:
            if item['result'] == gvalue:
                result.append(item['value'])
                
        return result
    
    def __calculate_root_rule_truth_table(self, rule: str, gvalue: str) -> List:
        fact_num = rule.count('[F]')
        # replace [F] with placeholder fact
        for i in range(fact_num):
            rule = rule.replace("[F]", self.fact_holder[i], 1)
        rule = f"∀x ({rule})"
        rule = self.converter.convert_expression(rule)
        
        # get truth table with the help of theorem prover
        value_table = list(product(['True', 'False', 'Uncertain'], repeat=fact_num))
        truth_table = []
        for item in value_table:
            arguments = [((rule), self.assign_rule_value(item, fact_list=self.fact_holder[:fact_num]))]
            
            prover9_result = self.prover9(arguments_list=arguments, some_is_goal=False)
            truth_table.append(
                {
                    'value': item,
                    'result': prover9_result
                }
            )
        
        result = []
        for item in truth_table:
            if item['result'] == gvalue:
                result.append(item['value'])
                
        return result
        
    def load_rule_candidate(self, file_path):
        # load rules from the file
        rule_candidate = self.load_json(file_path)
        self.normal_rules = rule_candidate['normal_rules']
        self.goal_rules = rule_candidate['goal_rules']
        
    @staticmethod
    def prover9(arguments_list: List, some_is_goal: bool = False) -> str:
        result1 = prover9_prove(arguments_list)
        
        assert "all x" in arguments_list[0][0]
        if "all x" in arguments_list[0][0]:
            false_conclusion = f"all x. (not ({arguments_list[0][0].replace('all x. ', '')}))"
        else:
            false_conclusion = f"not ({arguments_list[0][0]})"
        
        false_arguments = [(false_conclusion, arguments_list[0][1])]
        result2 = prover9_prove(false_arguments)
        
        if result1 == result2:
            return "Uncertain"
        elif result1 == True and result2 == False:
            return "True"
        else:
            return "False"
    
    @staticmethod
    def assign_rule_value(value: Tuple, fact_list: List) -> List[str]:
        result= []
        for i in range(len(value)):
            fact = fact_list[i]
            if value[i] == 'True':
                result.append(f"all x. ({fact})")
            elif value[i] == 'False':
                result.append(f"all x. (not ({fact}))")
            elif value[i] == 'Uncertain':
                continue
            else:
                raise ValueError(f"Unsupported value: {value[i]}")
            
        return result
    
    @staticmethod
    def load_json(file_path) -> Dict:
        with open(file_path, 'r') as f:
            result = json.load(f)
            
        return result