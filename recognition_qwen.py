import base64
import os
from openai import OpenAI
from tqdm import tqdm
from random import shuffle
import json

# 读取data下所有文件夹的名称和其中的T.png
dirs = os.listdir("data")

T_img_dicts = {}
for d in dirs:
    files = os.listdir("data/"+d)
    with open("data/"+d+"/T.png", 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')
    T_img_dicts[d] = img_base

# 读取data下所有文件夹中的信息.json
info_dicts = {}
for d in dirs:
    info_path = "data/"+d+"/信息.json"
    with open(info_path, 'r', encoding='utf-8') as f:
        info_dicts[d] = json.load(f)["描述"]


client = OpenAI(
    api_key="sk-a95d7ce4cdbf47d5b78a1ab155096cd8",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

print(len(T_img_dicts.keys()))
# print(len(D_img_dicts.keys()))
print(len(info_dicts.keys()))

names = list(T_img_dicts.keys())

# info_list = list(info_dicts.values())


def prompt_to_messages(img, info):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "你是一个专业的游戏角色描述助手，你需要根据图片和全部角色信息，判断图中是哪位角色。"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img}"
                    }
                },
                {
                    "type": "text",
                    "text": "请分析图片得到其中中心角色的外观描述，判断最符合全部角色中的哪位角色的描述：\n" + str(info) + "\n只返回角色名"
                }
            ]
        }
    ]


result = {}
for name, T_img in tqdm(T_img_dicts.items()):
    shuffle(names)
    shuffled_dict = {key: info_dicts[key] for key in names}
    response = client.chat.completions.create(
        model="qwen2.5-vl-32b-instruct",  # 填写需要调用的模型名称
        messages=prompt_to_messages(T_img, shuffled_dict)
    )
    result[name] = response.choices[0].message.content

print(result)
