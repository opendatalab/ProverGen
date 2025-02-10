import os
import json
import random
import pickle
import argparse
import time
import logging

import numpy as np

from utils.logic_translator.translator import Translator
from utils.logic_translator.noise import NoiseTranslator
from utils.logic_translator.generator import ProblemGenerator


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logic_skeleton_translator_{time.strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser()
    
    # required parameters
    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--mode", type=str, default='hard')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=300)
    parser.add_argument("--data_dir", type=str, default="outputs/logic_data")
    parser.add_argument("--output_dir", type=str, default="outputs/translated_data")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    
    # For local models
    parser.add_argument("--base_url", type=str, default="EMPTY")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    
    # default parameters
    parser.add_argument("--predicate_path", type=str, default="data/wordnet_predicates.json")
    parser.add_argument("--example_path", type=str, default="data/translation_examples.json")
    parser.add_argument("--name_path", type=str, default="data/names")
    parser.add_argument("--seed", type=int, default=727)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    logger.info(f"Starting logic skeleton translation with args: {vars(args)}")
    
    seed_everything(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    start_time = time.time()
    
    # load dataset
    logger.info(f"Loading dataset from {args.data_dir}/{args.mode}-{args.num}.pickle")
    with open(f'{args.data_dir}/{args.mode}-{args.num}.pickle', 'rb') as f:
        logic_data = pickle.load(f)
        
    # translate facts and rules
    logger.info("Initializing translator and starting facts and rules translation...")
    translator = Translator(args)
    translated_problems = translator.translate_rules_and_facts(data=logic_data)
    
    # translate distracting facts and rules
    logger.info("Translate distracting facts and rules...")
    noise_translator = NoiseTranslator(args, translated_data=translated_problems)
    translated_problems = noise_translator.create_distracting_rules()
    
    # generate problems
    logger.info("Generating final problems...")
    problem_generator = ProblemGenerator(args, translated_data=translated_problems)
    translated_problems = problem_generator.create_problems()
    
    # save result
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Created output directory: {args.output_dir}")
    
    output_path = f"{args.output_dir}/{args.mode}-{args.num}-{args.start}_{args.end}.json"
    logger.info(f"Saving translated problems to {output_path}")
    with open(output_path, "w") as f:
        json.dump(translated_problems, f, indent=2, ensure_ascii=False)
        
    duration = time.time() - start_time
    logger.info(f"Total time: {duration:.2f} seconds")
    logger.info(f"Average time per problem: {duration / args.num:.2f} seconds")
    
    logger.info("Logic skeleton translation completed successfully.")
    
