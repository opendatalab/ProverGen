import os
import re
import json
import time
import random
import argparse
from typing import List
import logging

import numpy as np

from utils.augmentor import DataAugmentor
        

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def load_names(file_dir: str) -> List:
    name_list = []
    filenames = os.listdir(file_dir)

    for filename in filenames:
        filepath = os.path.join(file_dir, filename)
        with open(filepath, 'r') as f:
            c_name_list = json.load(f)
            
        c_name_list = c_name_list['names']
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


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'fol_problem_generator_{time.strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--mode", type=str, default="uncertain_augment", help="the generation mode, can be one of normal_generation, step_augment, uncertain_augment")
    parser.add_argument("--filepath", type=str, default="outputs/translated_data/hard-300-0_1.json")
    parser.add_argument("--output_dir", type=str, default="outputs/final_data")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    
    # For local models
    parser.add_argument("--base_url", type=str, default="EMPTY")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    
    parser.add_argument("--predicate_file", type=str, default="data/wordnet_predicates.json")
    parser.add_argument("--noise1", type=float, default=1, help="Type I distraction")
    parser.add_argument("--noise2", type=float, default=1, help="Type II distraction")
    parser.add_argument("--seed", type=int, default=727)
    parser.add_argument("--name_path", type=str, default="data/names")
    
    args = parser.parse_args()
    
    logger.info(f"Starting fol problem generation with args: {vars(args)}")

    seed_everything(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    # generate dataset
    name_list = load_names(args.name_path)
    with open(args.filepath, 'r') as f:
        loaded_data = json.load(f)
    
    logger.info(f"Generating {len(loaded_data)} problems in {args.mode} mode...")
    augmentor = DataAugmentor(args=args)
    if args.mode == "step_augment":  # Break down each step of reasoning and present it as a new question.
        augmented_data = augmentor.step_augment(data=loaded_data, shuffled=True, has_noise1=args.noise1, has_noise2=args.noise2, name_list=name_list, start=args.start, end=args.end)
    elif args.mode == "uncertain_augment":  # Break down each step of reasoning to generate a new problem with uncertain answer
        augmented_data = augmentor.uncertain_augment(data=loaded_data, shuffled=True, has_noise1=args.noise1, has_noise2=args.noise2, name_list=name_list, start=args.start, end=args.end)
    elif args.mode == "normal_generation":  # Modify the data according to the common format.
        augmented_data = augmentor.normal_generation(data=loaded_data, shuffled=True, has_noise1=args.noise1, has_noise2=args.noise2, name_list=name_list, start=args.start, end=args.end)
        args.mode = "test"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Created output directory: {args.output_dir}")
        
    output_path = f"{args.output_dir}/{args.mode}-{args.start}_{args.end}.json"
    logger.info(f"Saving generated problems to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(augmented_data, f, indent=2, ensure_ascii=False)
    
    
        
        