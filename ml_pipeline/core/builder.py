from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from typing import Union

from .manager import exists_estimator, create_estimator


class PipelineBuilder(object):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def build(self) -> Pipeline:
        return self._parse(self.config)

    def _parse(self, config: Union[dict, list]) -> Union[dict, list, BaseEstimator]:        
        if isinstance(config, dict):

            if (len(config) == 1 
                and exists_estimator(list(config)[0])):
                name = list(config)[0]
                return create_estimator(name,
                                        **self._parse(config[name]))
            
            params = dict()
            for name in config:
                
                value = config[name]
                if isinstance(value, dict):
                    params[name] = self._parse(value)
                elif isinstance(value, list):
                    params[name] = [self._parse(v) for v in value]
                else:
                    if exists_estimator(value):
                        params[name] = create_estimator(value, **dict())
                    else:
                        params[name] = value.rstrip('_') if isinstance(value, str) else value
            return params

        elif isinstance(config, list) or isinstance(config, tuple):

            params = list()
            for value in config:
                if isinstance(value, dict):
                    params.append(self._parse(value))
                elif isinstance(value, list):
                    params.append([self._parse(v) for v in value])
                else:
                    if exists_estimator(value):
                        params.append(create_estimator(value, **dict()))
                    else:
                        params.append(value.rstrip('_') if isinstance(value, str) else value)
            return params

        else:
            raise AttributeError(config)
