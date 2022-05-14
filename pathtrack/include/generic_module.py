import sys
import yaml
import json
import numpy as np

# Load yaml and return data as a dictionary object
def loadyaml(yaml_path):
    try:
        with open(yaml_path) as file:
            obj = yaml.safe_load(file)
            return obj
    except Exception as e:
        print('Exception occurred while loading YAML...', file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

# identify json format
def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError as e:
        return False
    return True