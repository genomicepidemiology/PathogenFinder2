import os
import shutil


def read_multifiles(path):
    list_files = []
    list_basefile = []
    with open(path, "r") as path_read:
        for line in path_read:
            list_files.append(line.rstrip())
            list_basefile.append(os.path.basename(line.rstrip()))
    return list_files, list_basefile


def center_print(msg: str) -> str:
    return msg.center(shutil.get_terminal_size().columns)

