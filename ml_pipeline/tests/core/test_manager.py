import sys
import pytest

from sklearn.impute import SimpleImputer

sys.path.append('.')
from ml_pipeline.core import manager


def test_exist_and_create() -> None:
    manager.register_estimator(SimpleImputer)

    assert manager.exists_estimator('simple_imputer')
    assert manager.exists_estimator('ababababa') == False

    estimator = manager.create_estimator('simple_imputer', strategy='mean')
    assert estimator is not None
