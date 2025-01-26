from munch import Munch
import yaml

def load_config(path):
    with open(path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Munch(config_dict)