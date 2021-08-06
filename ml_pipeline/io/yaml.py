import yaml


def read_yaml(path: str):
    with open(path, 'r') as file:
        content = yaml.safe_load(file)
    return content
