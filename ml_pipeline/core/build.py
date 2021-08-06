from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .manager import createEstimator


class PipelineBuilder(object):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def build(self) -> Pipeline:
        if not isinstance(self.config['pipeline'], list):
            raise ValueError
        return self._parse_pipeline(self.config['pipeline'])

    @staticmethod
    def _parse_pipeline(items: list) -> Pipeline:
        pipeline = []
        for item in items:
            if 'pipeline' in item:
                pipeline.append([item['name'],
                                 PipelineBuilder._parse_pipeline(item['pipeline'])])
            elif 'estimator' in item:
                pipeline.append([item['name'],
                                 PipelineBuilder._parse_estimator(item['estimator'],
                                                                  item.get('params', dict()))])
            else:
                raise ValueError
        return Pipeline(steps=pipeline)

    @staticmethod
    def _parse_estimator(type: str, params: dict) -> BaseEstimator:
        return createEstimator(type,
                               **PipelineBuilder._parse_params(params))

    @staticmethod
    def _parse_params(params: dict) -> dict:
        for key, val in params.items():
            if (isinstance(val, dict) 
                and PipelineBuilder._is_estimator(val)):
                params[key] = PipelineBuilder._parse_estimator(val['estimator'],
                                                               val.get('params', dict()))
            elif isinstance(val, list):
                values = []
                for v in val:
                    if (isinstance(val, dict) 
                        and PipelineBuilder._is_estimator(v)):
                        values.append(PipelineBuilder._parse_estimator(v['estimator'],
                                                                       v.get('params', dict())))
                    else:
                        values.append(v)
                params[key] = values
        return params

    @staticmethod
    def _is_estimator(config: dict) -> bool:
        if (len(config) == 2
            and 'estimator' in config.keys() 
            and 'params' in config.keys()
            and isinstance(config['params'], dict)):
            return True
        elif (len(config) == 1
              and 'estimator' in config.keys()
              and not isinstance(config['estimator'], dict)):
            return True
        else: 
            return False

