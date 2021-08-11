import sys
import pytest

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import Normalizer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectFromModel

sys.path.append('.')
from ml_pipeline.core import manager
from ml_pipeline.core.builder import PipelineBuilder


manager.register_estimator(Pipeline)
manager.register_estimator(ColumnTransformer)
manager.register_estimator(make_column_selector)
manager.register_estimator(Normalizer)
manager.register_estimator(SimpleImputer)
manager.register_estimator(RandomForestClassifier)
manager.register_estimator(VarianceThreshold)
manager.register_estimator(SelectFromModel)


def test_build() -> None:
    transformer_cfg = {'column_transformer': {'transformers': [['norm',
                                                                {'normalizer': {'norm': 'l1'}},
                                                                {'make_column_selector': {'pattern': '*'}},]],
                                              'remainder': 'passthrough',}}

    feature_importance_cfg = [['fill_na', {'simple_imputer': {'strategy': 'mean'}}],
                              ['select_from_model_', {'select_from_model': {'estimator': {'random_forest_classifier': {'n_estimators': 20}},
                                                                            'threshold': 1e-2}}]]

    selector_cfg = {'pipeline': {'steps': [['variance', {'variance_threshold': {'threshold': 1e-2}}], 
                                           ['feature_importance', {'pipeline': {'steps': feature_importance_cfg}}]]}}

    config = {'pipeline': {'steps': [('transformer', transformer_cfg),
                                     ('selector', selector_cfg),
                                     ('model', 'random_forest_classifier')]}}

    builder = PipelineBuilder(config)
    pipeline = builder.build()

    assert pipeline is not None
    assert len(pipeline) == 3
    assert isinstance(pipeline[0], ColumnTransformer)
    assert isinstance(pipeline[0].transformers[0][1], Normalizer)
    assert isinstance(pipeline[1], Pipeline)
    assert isinstance(pipeline[1][0], VarianceThreshold)
    assert isinstance(pipeline[1][1], Pipeline)
    assert isinstance(pipeline[1][1][0], SimpleImputer)
    assert isinstance(pipeline[1][1][1], SelectFromModel)
    assert isinstance(pipeline[1][1][1].estimator, RandomForestClassifier)
    assert isinstance(pipeline[2], RandomForestClassifier)
