import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab



# 读取文件内容的函数
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip().split('\n')


# 读取数据
data_A = read_file('before_edit.txt')
data_B = read_file('edit_instruct.txt')
data_C = read_file('after_edit.txt')

if not (len(data_A) == len(data_B) == len(data_C)):
    raise ValueError("数据行数不一致，请确保三个文件中的行数相同")

# 将数据组织成一个列表，每个元素是一个字典
data_list = [
    {
        '编辑前的描述': a,
        '编辑指令': b,
        '编辑后的描述': c
    }
    for a, b, c in zip(data_A, data_B, data_C)
]

# 将列表写入 JSON 文件
with open('data.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, ensure_ascii=False, indent=4)

print("数据已成功写入到 data.json 文件")