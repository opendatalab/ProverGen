from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Problems:
    id: int
    goal: Dict
    facts: List
    rules: List
    reasoning_chain: List
    goal_nl: str = None
    translated_facts: List = None