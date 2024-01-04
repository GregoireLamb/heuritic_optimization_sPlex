from abc import ABC, abstractmethod

from ConfigSpace import Configuration
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter, \
    UniformIntegerHyperparameter

from src.hyperparam_opti.hyperparam_config import Config


class TargetFunctionWrapper(ABC):
    """
    Abstract class for target function wrappers. Implementations of this class should implement:
    - evaluate: Evaluate the function with the given configuration
    - _load_param_space: Return the parameter space
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._config = config
        self._param_space = self._load_param_space()

    @abstractmethod
    def evaluate(self, config: Configuration) -> float:
        """
        Evaluate the function with the given configuration
        """
        pass

    def get_param_space(self) -> ConfigurationSpace:
        """
        Return the parameter space
        """
        return self._param_space

    def _load_param_space(self) -> ConfigurationSpace:
        """
        Return the parameter space
        """
        conf_space = ConfigurationSpace(name='BoltHyperparameters')

        parameters = []
        for name, settings in self._config.hyperparameters.items():
            if 'include' in settings and not settings['include']:
                continue
            if settings['type'] == 'float':
                param = UniformFloatHyperparameter(name=name, lower=settings['min'], upper=settings['max'],
                                                   default_value=settings.get('default', 0.5 * (
                                                               settings['min'] + settings['max'])))
            elif settings['type'] == 'int':
                param = UniformIntegerHyperparameter(name=name, lower=settings['min'], upper=settings['max'],
                                                     default_value=settings.get('default', int(0.5 * (
                                                                 settings['min'] + settings['max']))))
            elif settings['type'] == 'bool':
                default = 'true' if settings.get('default', False) else 'false'
                param = CategoricalHyperparameter(name=name, default_value=default, choices=['true', 'false'])
            elif settings['type'] == 'categorical':
                param = CategoricalHyperparameter(name=name, choices=settings['options'],
                                                  default_value=settings.get('default', settings['options'][0]))
            else:
                raise ValueError(f'Parameter type {settings["type"]} not supported')
            parameters.append(param)

        for param in parameters:
            conf_space.add_hyperparameter(param)

        return conf_space