import sys
import pytest

sys.path.append('.')
from ml_pipeline.io import yaml


def test_yaml_read() -> None:
    true_config = {'pipeline': {'steps': [['transformer', {'column_transformer': {'transformers': [['norm',
                                                                                                    {'normalizer': {'norm': 'l1'}},
                                                                                                    {'make_column_selector': {'pattern': '*'}},]],
                                                                                  'remainder': 'passthrough',}}]]}}
    config = yaml.read_yaml('ml_pipeline/tests/io/config.yaml')

    assert config == true_config