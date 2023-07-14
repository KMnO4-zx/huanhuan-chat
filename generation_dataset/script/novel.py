import re
import sys
import os
import chardet


def decode_file_content(file_content):
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'big5']

    for encoding in encodings:
        try:
            decoded_content = file_content.decode(encoding)
            return decoded_content
        except UnicodeDecodeError:
            pass

    raise UnicodeDecodeError(
        "All tested encodings failed", file_content, 0, 0, "")


def split_chapters(input_file):
    with open(input_file, 'rb') as f:
        file_content = f.read()

    content = decode_file_content(file_content)

    pattern = r"(第[\u4E00-\u9FA5]+章[，\s]+|[\d]+[\u4E00-\u9FA5]+)"
    chapters = re.split(pattern, content)

    output_dir = "../raw/novel"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(1, len(chapters), 2):
        chapter_file = os.path.join(output_dir, f"{chapters[i]}.txt")

        with open(chapter_file, 'w', encoding='utf-8') as f:
            f.write(chapters[i] + '\n')
            f.write(chapters[i + 1])


if __name__ == '__main__':
    split_chapters('后宫—甄嬛传.txt')
