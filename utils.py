import jsonlines

def save_arr(arr, save_path):
    with jsonlines.open(save_path, "w") as writer:
        for d in arr:
            writer.write(d)

def read_jsonl(data_path):
    contents = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            contents.append(obj)
    return contents
