from importlib.resources import files

import yaml


RESOURCE_DIR = files("bot_or_not").joinpath("resources")
CONFIG_PATH = RESOURCE_DIR.joinpath("config.yaml")

with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)


APP_NAME = config["app"]["name"]
APP_VERSION = config["app"]["version"]
APP_DESCRIPTION = config["app"]["description"]

MODEL_FEATURES = config["model"]["features"]
NHT_CLASS_LABEL = config["model"]["positive_class"]

RANDOM_STATE = config["training"]["random_state"]
FALSE_POSITIVE_COST = config["evaluation"]["false_positive_cost"]
FALSE_NEGATIVE_COST = config["evaluation"]["false_negative_cost"]

MODEL_PATH = RESOURCE_DIR.joinpath(
    config["paths"]["model"]
)