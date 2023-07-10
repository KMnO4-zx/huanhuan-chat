import json
import re


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines


def parse_dialogue(dialogue, lines):
    for i in range(1, len(lines)-1):
        mline = lines[i]
        if mline.startswith('甄嬛：') == False:
            continue
        mline = mline.replace("（", "(")
        mline = mline.replace("）", ")")
        mline = mline.replace("：", ":")
        mline = mline.replace("\n", "")
        line = re.sub(r'\([^)]*\)', '', mline)
        mlast_line = lines[i-1]
        mlast_line = mlast_line.replace("（", "(")
        mlast_line = mlast_line.replace("）", ")")
        mlast_line = mlast_line.replace("：", ":")
        mlast_line = mlast_line.replace("\n", "")
        print(mlast_line)
        print(mline)
        last_line = re.sub(r'\([^)]*\)', '', mlast_line)
        if re.match(r'^\w+:+?', line) and re.match(r'^\w+:+?', last_line):
            character = last_line.split(':')[0]
            p_dialogue = last_line.split(':')[1]
            l_dialogue = line.split(':')[1]
            tmp = {
                "instruction": p_dialogue,
                "input": "",
                "output": l_dialogue
            }
            dialogue.append(tmp)
    return dialogue


def write_json(dialogue, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(dialogue, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    files = ['甄嬛传剧本01-10.txt', '甄嬛传剧本11-20.txt', '甄嬛传剧本21-30.txt', '甄嬛传剧本31-40.txt',
             '甄嬛传剧本41-50.txt', '甄嬛传剧本51-60.txt', '甄嬛传剧本61-70.txt', '甄嬛传剧本71-76.txt']
    dialogue = []
    path = '../raw/'
    for file_name in files:
        file_path = path + file_name
        lines = read_file(file_path)
        parse_dialogue(dialogue, lines)
    zhenhuan = {
        "instruction": "你是谁？",
        "input": "",
        "output": "我是甄嬛，家父是大理寺少卿甄远道。"
    }
    for _ in range(50):
        dialogue.append(zhenhuan)
    write_json(dialogue, 'zhenhuan.json')
    print(len(dialogue))
