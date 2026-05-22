import os
import json


def ensure_directory_exists(file_path):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)


def write_json(file_path, data):
    ensure_directory_exists(file_path)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def write_jsonl(file_path, data_list):
    ensure_directory_exists(file_path)
    with open(file_path, 'w') as f:
        for data in data_list:
            f.write(json.dumps(data) + "\n")


def write_jsonl2(file_path, data):
    ensure_directory_exists(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def append_jsonl(file_path, data):
    ensure_directory_exists(file_path)
    with open(file_path, "a", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False) + "\n"
            f.write(json_line)


def read_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}


def read_jsonl(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
    return []
