from datetime import datetime
from typing import Union

from smac import Scenario
from src.hyperparam_opti.hyperparam_config import Config
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt

import json

from smac.main.smbo import SMBO
from smac.facade import HyperparameterOptimizationFacade as HPOFacade
from function_wrapper import TargetFunctionWrapper
from instances import Instances
from smac.runhistory import TrialInfo, TrialValue, StatusType
from smac import Callback

GENERAL_LIMIT = 100000000


class StopCallback(Callback):
    def __init__(self, early_iteration_limit: int, early_walltime_limit: int, config: Config):
        self._early_iteration_limit = early_iteration_limit
        self._early_walltime_limit = early_walltime_limit

        self._n_iterations_without_improvement = config.n_iterations_without_improvement
        self._n_workers = config.n_workers

    def on_tell_end(self, smbo: SMBO, info: TrialInfo, value: TrialValue) -> Union[bool, None]:
        """
        Called after the stats are updated and the trial is added to the runhistory.
        Optionally, returns false to gracefully stop the optimization.

        We stop the optimization if we reach the early iteration limit or the early walltime limit.
        """
        if smbo.runhistory.finished == self._early_iteration_limit:
            logging.info(
                f'Early stopping because we reached the early iteration limit of {self._early_iteration_limit} trials')
            return False

        if GENERAL_LIMIT - smbo.remaining_walltime > self._early_walltime_limit:
            logging.info(
                f'Early stopping because we reached the early walltime limit of {self._early_walltime_limit} seconds')
            return False

        if self._early_stopping(smbo):
            logging.info(
                f'Early stopping because we reached the early stopping criteria: {self._n_iterations_without_improvement} iterations without improvement')
            return False

        return None

    def _early_stopping(self, smbo: SMBO) -> bool:
        # TODO: Implement early stopping criteria

        history_df = get_history_df(smbo.runhistory)

        # Filter out StatusType != SUCCESS
        history_df = history_df[history_df['status'] == StatusType.SUCCESS]

        inc_id = smbo.runhistory.get_config_id(smbo.intensifier.get_incumbent())
        try:
            inc_index = history_df[history_df['config_id'] == inc_id].index[0]
        except IndexError:
            return False
        # Up to n_workers - 1 configurations could be explored without the information of the incumbent, so we not count them as non-improvements
        if len(history_df) - inc_index - 1 >= self._n_iterations_without_improvement + self._n_workers - 1:
            return True

        return False


class HyperparamOptimizer:
    """
    Hyperparameter Optimizer
    Uses SMAC to optimize the target function over the given set of instances
    """

    def __init__(self, config: Config, wrapper: TargetFunctionWrapper, instances: Instances):
        self._config = config
        self._function_wrapper = wrapper
        self._instances = instances
        assert len(self._instances.instances) <= 1, 'Multiple instances not supported yet'

        self._param_space = self._function_wrapper.get_param_space()
        self._smac = None

    def run(self):
        """
        Run the Hyperparameter Optimizer
        """

        if self._config.start_new_run:
            self._config.execution_name = f'{self._config.execution_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            logging.info(f'Starting new run. Execution name: {self._config.execution_name}')
        else:
            logging.info(f'Trying to continue a previous run with name {self._config.execution_name}')

        scenario = Scenario(
            configspace=self._param_space,
            name=f'{self._config.execution_name}',
            deterministic=True,
            use_default_config=True,
            n_trials=GENERAL_LIMIT,
            walltime_limit=GENERAL_LIMIT,
            instances=self._instances.instances if len(self._instances.instances) > 1 else None,
            instance_features=self._instances.instance_features if len(self._instances.instances) > 1 else None,
            n_workers=self._config.n_workers,
            output_directory=self._config.smac_output_dir,
        )

        stop_callback = StopCallback(self._config.n_trials, self._config.walltime_limit, self._config)

        self._smac = HPOFacade(
            scenario=scenario,
            target_function=self._function_wrapper.evaluate,
            callbacks=[stop_callback],
            overwrite=self._config.start_new_run)

        if not self._config.only_get_results:
            incumbent = self._smac.optimize()
            logging.info(f'Optimization finished. Incumbent: {incumbent}')
        else:
            incumbent = self._smac.intensifier.get_incumbent()
            logging.info(f'Incumbent retrieved: {incumbent}')

        self._save_incumbent()
        self._analyze_run_history()

    def _analyze_run_history(self):
        """
        Analyze the run history of the optimization
        """
        runhistory = self._smac.runhistory
        output_dir = self._smac.scenario.output_directory

        history_df = get_history_df(runhistory)

        configs_df, optimal_id = self._aggregate_over_instances(runhistory, history_df)

        # Save the run history as csv
        configs_df.to_csv(f'{output_dir}/ruhistory_{self._config.execution_name}.csv', index=False)

        # Save configurations that are up to 15% worse than the incumbent
        top_configs_df = configs_df[configs_df['average_cost'] <= 1.15 *
                                    configs_df[configs_df['config_id'] == optimal_id]['average_cost'].values[0]]
        top_configs = sorted([({'cost': x.average_cost}, runhistory.get_config(x.config_id).get_dictionary()) for x in
                              top_configs_df.itertuples()], key=lambda x: x[0]['cost'])
        with open(f'{output_dir}/top_configs_{self._config.execution_name}.json', 'w') as f:
            json.dump(top_configs, f, indent=4)

        # Plot the evolution of the costs over time
        self._plot_costs_over_time(history_df)

    def _plot_costs_over_time(self, history_df):
        history_df = history_df[history_df['status'] == StatusType.SUCCESS].copy()
        history_df.sort_values(by=['starttime'], inplace=True)
        history_df['incumbent_cost'] = history_df['cost'].expanding().min()
        history_df['average_history_cost'] = history_df['cost'].expanding().mean()
        history_df['n_ma'] = history_df['cost'].rolling(window=self._config.n_workers * 2).mean()

        plt.figure(figsize=(10, 6))

        plt.plot(history_df.index, history_df['cost'], label='Cost')
        plt.plot(history_df.index, history_df['incumbent_cost'], label='Incumbent Cost')
        # plt.plot(history_df.index, history_df['average_history_cost'], label='Average History Cost')
        plt.plot(history_df.index, history_df['n_ma'], label=f'{self._config.n_workers * 2} - Moving Average')

        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Evolution of Cost and Incumbent Cost')
        plt.legend()

        # Save the plot as png
        output_dir = self._smac.scenario.output_directory
        plt.savefig(f'{output_dir}/costs_over_time_{self._config.execution_name}.png')

    def _aggregate_over_instances(self, runhistory, history_df):
        configs_df = history_df.groupby(['config_id'], as_index=False).agg(
            first_run=('starttime', 'min'),
            last_run=('endtime', 'max'),
            n_runs=('starttime', 'count'),
        )
        optimal_id = runhistory.get_config_id(self._smac.intensifier.get_incumbent())
        configs_df['average_cost'] = configs_df['config_id'].apply(
            lambda x: runhistory.average_cost(runhistory.get_config(x)))
        configs_df['optimal_conf'] = configs_df['config_id'].apply(lambda x: 1 if x == optimal_id else 0)
        return configs_df, optimal_id

    def _save_incumbent(self):
        """
        Save the results of the optimization
        """
        output_dir = self._smac.scenario.output_directory

        # Save the best configuration as json
        with open(f'{output_dir}/config_{self._config.execution_name}.json', 'w') as f:
            json.dump(self._smac.intensifier.get_incumbent().get_dictionary(), f, indent=4)


def get_history_df(runhistory: SMBO.runhistory):
    """
    Converts the given runhistory object into a pandas DataFrame.

    Args:
        runhistory: The SMAC runhistory object containing the data.

    Returns:
        history_df: A pandas DataFrame containing the converted data.
    """
    history_df = pd.DataFrame(
        columns=['config_id', 'instance', 'seed', 'cost', 'time', 'status', 'starttime', 'endtime'])
    for k, v in runhistory._data.items():
        row_values = list({
                              'config_id': int(k.config_id),
                              'instance': str(k.instance),
                              'seed': k.seed if k.seed is not None else None,
                              'cost': float(v.cost),
                              'time': v.time,
                              'status': v.status,
                              'starttime': pd.to_datetime(v.starttime, unit='s'),
                              'endtime': pd.to_datetime(v.endtime, unit='s')
                          }.values())
        history_df.loc[len(history_df), :] = row_values
    return history_df
