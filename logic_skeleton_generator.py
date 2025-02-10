import os
import random
import pickle
import argparse
import time
import logging

import numpy as np

from utils.logic_generator.generator import LogicGenerator


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    

if __name__ == "__main__":
    # Set up the path of Prover9
    os.environ['PROVER9'] = 'LADR-2009-11A/bin'

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logic_skeleton_generator_{time.strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # required arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=300)
    parser.add_argument("--mode", type=str, default="hard")
    parser.add_argument("--output_dir", type=str, default="outputs/logic_data")
    
    # default arguments
    parser.add_argument("--seed", type=int, default=730)
    parser.add_argument("--goal_value_probs", type=str, default="[1/3, 1/3, 1/3]", help="The proportion of True, False and Uncertain. This value should be given in standard list format, such as [0.4, 0.3, 0.3]")
    parser.add_argument("--rule_candidate_path", type=str, default="data/rules.json")
    parser.add_argument("--rule_as_goal_proportion", type=str, default="[0.75, 0.25]", help="The first number represents the proportion of logic skeletons with a fact conclusion, while the second indicates those with a rule conclusion.")
    parser.add_argument("--fact_num_threshold", type=int, default=2, help="when the size of the fact pool exceeds the threshold, there will be a possibility that the fact will be directly given")
    parser.add_argument("--fact_num_prob", type=float, default=0.4)
    
    args = parser.parse_args()
    args.goal_value_probs = eval(args.goal_value_probs)
    args.rule_as_goal_proportion = eval(args.rule_as_goal_proportion)
    
    logger.info(f"Starting logic skeleton generation with args: {vars(args)}")
    
    # set random seed
    seed_everything(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # start generation
    start_time = time.time()
    logger.info("Initializing logic skeleton generator...")
    problem_generator = LogicGenerator(args)
    logger.info(f"Generating {args.num} problems in {args.mode} mode...")
    problems = problem_generator.generate_logic_skeletons(verbose=False)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Created output directory: {args.output_dir}")
    
    output_path = f"{args.output_dir}/{args.mode}-{args.num}.pickle"
    logger.info(f"Saving generated problems to {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(problems, f)

    duration = time.time() - start_time
    logger.info(f"Total time: {duration:.2f} seconds")
    logger.info(f"Average time per problem: {duration / args.num:.2f} seconds")
    
    logger.info("Logic skeleton generation completed successfully.")

    
    
    
    
        