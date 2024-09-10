import json
import os


origin_path = 'data.json'
new_path = 'train.jsonl'



def dataset_jsonl_transfer(origin_path, new_path):

    messages = []

    with open(origin_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            before_edit = item["编辑前的描述"]
            edit_instruct = item["编辑指令"]
            after_edit = item["编辑后的描述"]
            message = {
                "instruction": "你是一个写编辑指令的专家，你会接收到一句话,这句话是编辑前的描述，请输出你要做的编辑指令和编辑之后的描述",
                "input": f"编辑前的描述:{before_edit}",
                "output": f"编辑指令：{edit_instruct},编辑后的描述：{after_edit}",
            }
            messages.append(message)

        with open(new_path, "w", encoding="utf-8") as file:
            for message in messages:
                file.write(json.dumps(message, ensure_ascii=False) + "\n")


if not os.path.exists(new_path):
    dataset_jsonl_transfer(origin_path, new_path)









