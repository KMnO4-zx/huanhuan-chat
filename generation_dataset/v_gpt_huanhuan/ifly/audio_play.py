import os

def play(file_name):
    """audio play"""
    os.system(f"ffplay -autoexit -hide_banner {file_name}")		#播放完毕自动退出

