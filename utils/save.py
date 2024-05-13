import json


def json_dump(path,file_to_save,type="w+"):
    with open(path,type) as f:
        json.dump(file_to_save,f)