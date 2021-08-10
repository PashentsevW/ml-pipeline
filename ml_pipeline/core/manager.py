from sklearn.base import BaseEstimator
from ..utils import camel_to_snake

_estimators = {}


def register_estimator(class_type: type) -> None:
    class_name = class_type.__name__
    _estimators[camel_to_snake(class_name)] = class_type


def exists_estimator(name: str) -> bool:
    return _estimators.get(name) is not None


def create_estimator(name: str, **params) -> BaseEstimator:
    return _estimators[name](**params)
