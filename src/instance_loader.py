from src.config import Config


class InstanceLoader:
    def __init__(self, config: Config):
        self._config = config

    def load(self, path):
        """
        Load instances according to config instance type
        """
        pass
