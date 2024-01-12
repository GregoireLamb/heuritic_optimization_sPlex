from optimizer import HyperparamOptimizer
from hyperparam_config import Config
from function_tuning import FunctionTuning
from instances import Instances
import argparse
from datetime import datetime
import os

import logging

format = '[%(levelname)s][%(asctime)s][%(module)s:%(lineno)d] %(message)s'
logging.basicConfig(format=format, level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execution_type', type=str, default='single', choices=['single', 'multiple'],
                        help='Single runs a single SMAC execution according to config.yaml. Multiple runs multiple SMAC executions according to config_multiple.yaml')
    parser.add_argument('--worker_id', type=int, default=None, help='Worker id for multiple executions')
    args = parser.parse_args()
    if args.worker_id is not None and args.execution_type != 'multiple':
        raise ValueError('worker_id can only be used with execution_type multiple')

    config = Config(args.execution_type, args.worker_id)

    if config.exec_type == 'single':
        logging.info('Starting single SMAC execution')

        instances = Instances(config)
        wrapper = FunctionTuning(config, instances)

        bb_optimizer = HyperparamOptimizer(config, wrapper, instances)
        bb_optimizer.run()
