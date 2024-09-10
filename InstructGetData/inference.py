from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import json

mode_path = "./qwen/Qwen2-1___5B-Instruct/"
lora_path = "./output/Qwen1.5/checkpoint-60" # 这里改称你的 lora 输出对应 checkpoint 地址

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

prompt = "你是谁？"
instruction = "你是一个写编辑指令的专家，你会接收到一句话,这句话是编辑前的描述，请输出你要做的编辑指令和编辑之后的描述"
text_input = ["A big house.",
              "A big house.",
              "A big mountain.",
              "A big house.",
              "A big horse."]

'''
messages = [
        {"role": "system", "content": f"你是一个写编辑指令的专家，你会接收到一句话,这句话是编辑前的描述，请输出你要做的编辑指令和编辑之后的描述"},
        {"role": "user", "content": f"A big house."}
    ]
inputs = tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_tensors="pt",return_dict=True).to('cuda')
gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
'''

def get_data(str):
    key = '：'
    i = 0
    while str[i] != '：' :
        i += 1
    i += 1

    edit_instruction = ''
    while str[i] != ',':
        edit_instruction += str[i]
        i += 1

    while str[i] != '：':
        i += 1
    i += 1

    after_edit = ''
    while i != len(str):
        after_edit += str[i]
        i += 1

    return edit_instruction, after_edit


data_list = []

for data in text_input:
    messages = [
        {"role": "system",
         "content": instruction},
        {"role": "user", "content": data}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                           return_dict=True).to('cuda')
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        #print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        x = tokenizer.decode(outputs[0], skip_special_tokens=True)
        b, c = get_data(x)
        print(data, b, c)
        data_save = {"图像编辑前的描述" : data, "编辑指令" : b, "图像编辑后的描述" : c}
        data_list.append(data_save)


with open('data_generated.json', 'w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, ensure_ascii=False, indent=4)


