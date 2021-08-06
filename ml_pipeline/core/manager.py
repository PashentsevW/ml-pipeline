from sklearn.base import BaseEstimator
from ..utils import camel_to_snake

_estimators = {}


def registerEstimator(class_type: type) -> None:
    class_name = class_type.__name__
    _estimators[camel_to_snake(class_name)] = class_type


def createEstimator(name: str, **params) -> BaseEstimator:
    return _estimators[name](**params)
