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

    pattern = r"(第[\d]+幕)"
    chapters = re.split(pattern, content)
    output_dir = "../raw/drama_script"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(1, len(chapters), 2):
        chapter_file = os.path.join(output_dir, f"{chapters[i]}.txt")

        with open(chapter_file, 'w', encoding='utf-8') as f:
            f.write(chapters[i] + '\n')
            f.write(chapters[i + 1])


if __name__ == '__main__':
    files = ['甄嬛传剧本01-10.txt', '甄嬛传剧本11-20.txt', '甄嬛传剧本21-30.txt', '甄嬛传剧本31-40.txt',
             '甄嬛传剧本41-50.txt', '甄嬛传剧本51-60.txt', '甄嬛传剧本61-70.txt', '甄嬛传剧本71-76.txt']
    path = '../raw/'
    for file_name in files:
        file_path = path + file_name
        split_chapters(file_path)
