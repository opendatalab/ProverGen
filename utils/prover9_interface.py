from typing import List, Tuple

from nltk.sem.logic import Expression
from nltk.inference.prover9 import Prover9Command


def prover9_prove(arguments):
    """
    Try some proofs and exhibit the results.
    """
    for (goal, assumptions) in arguments:
        g = Expression.fromstring(goal)
        alist = [Expression.fromstring(a) for a in assumptions]
        p = Prover9Command(g, assumptions=alist).prove(verbose=False)
        
    return p

class FOL2Prover9Converter:
    """
    This is the converter to convert standard First Order Logic expression into the format that nltk prover9 accepts
    """
    
    def __init__(self) -> None:
        self.symbol2text = {
            "¬": "not ",
            "∃": "some ",
            "∀": "all ",
            "→": "->",
            "⟷": "<->",
            "∧": "&",
            "∨": "|",
            "↔": "<->",
        }
        self.lowercase_alphabet = [chr(i) for i in range(97, 123)]
        self.lowercase_alphabet += [str(i) for i in range(10)]
        
    def convert_fol_instance(self, assumption: List[str], goal: str) -> Tuple[List[str], str]:
        """convert a folio data instance.

        Args:
            assumption: a list of fol expression, including the facts and rules of current data instance
            goal: the fol expression that need to be proved True or False.

        Returns:
            the converted list of assumptions and goal
        """
        return [self.convert_expression(item) for item in assumption], self.convert_expression(goal)
        
        
    def convert_expression(self, fol_expression: str) -> str:
        """acceptes a first order logic as input and output the corresponding format

        Args:
            fol_expression (str): the standard first order logic expression

        Returns:
            str: converted version of the input expression
        """
        temp_expression = fol_expression
        for key in self.symbol2text:
            temp_expression = temp_expression.replace(key, self.symbol2text[key])
            
        # lower the character
        # tt = ""
        # for char in temp_expression:
        #     if char.isupper():
        #         tt += char.lower()
        #     else:
        #         tt += char
        
        temp_expression = temp_expression.replace("  ", " ")
        
        # modify "all" and "some"
        temp_expression = temp_expression.replace("all x all y all z", "all *x y z.")
        temp_expression = temp_expression.replace("all x all y", "all *x y.")
        temp_expression = temp_expression.replace("all x", "all x.")
        temp_expression = temp_expression.replace("*", "")
        
        temp_expression = temp_expression.replace("some x some y some z", "some *x y z.")
        temp_expression = temp_expression.replace("some x some y", "some *x y.")
        temp_expression = temp_expression.replace("some x", "some x.")
        temp_expression = temp_expression.replace("*", "")
        
        # modify ⊕
        if '⊕' in temp_expression:
            while temp_expression.find('⊕') != -1:
                symbol_index = temp_expression.find('⊕')
                
                # find left edge
                bracket_list = []
                left_edge = symbol_index - 1
                
                while left_edge > 0:
                    if temp_expression[left_edge] == ')':
                        bracket_list.append(')')
                    elif temp_expression[left_edge] == '(':
                        if len(bracket_list) == 0:
                            left_edge += 1
                            break
                        else:
                            assert bracket_list.pop() == ')'
                    left_edge -= 1
                
                # find right edge
                bracket_list = []
                right_edge = symbol_index + 1
                
                while right_edge < len(temp_expression):
                    if temp_expression[right_edge] == '(':
                        bracket_list.append('(')
                    elif temp_expression[right_edge] == ')':
                        if len(bracket_list) == 0:
                            break
                        else:
                            assert bracket_list.pop() == '('
                    right_edge += 1
                
                if (left_edge - 1 >= 0 and temp_expression[left_edge - 1] == '(') and (right_edge + 1 < len(temp_expression) and temp_expression[right_edge + 1] == ')'):
                    extracted_expression = temp_expression[left_edge - 1:right_edge + 1]
                    temp_expression = f"{temp_expression[:left_edge - 1]}(not {extracted_expression.replace('⊕', '<->', 1)}){temp_expression[right_edge + 1:]}"
                else:                
                    extracted_expression = temp_expression[left_edge:right_edge]
                    temp_expression = f"{temp_expression[:left_edge]}(not ({extracted_expression.replace('⊕', '<->', 1)})){temp_expression[right_edge:]}"
                    
        return temp_expression


