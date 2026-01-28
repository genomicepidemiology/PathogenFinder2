import os
import datetime
import shutil


def get_filename(file_path:str) -> (str,str):
    filebase = os.path.basename(file_path)
    ext = os.path.splitext(filebase)[1:]
    filename = os.path.splitext(filebase)[0]
    return filename, ext
    

def create_outputfolder(outpath:str) -> str:
 #   out_folder = '{outpath}_{date:%Y-%m-%d_%H-%M-%S}'.format(outpath=outpath,
#                                                               date=datetime.datetime.now())#
    out_folder = "{}".format(outpath)
    out_paths = {"main": out_folder,
                 "conf": "{}/configuration".format(out_folder)
                 }

    for k, v in out_paths.items():
        os.mkdir(v)

    return out_paths

def read_multifiles(path):
    list_files = []
    list_basefile = []
    with open(path, "r") as path_read:
        for line in path_read.readlines():
            list_files.append(line.rstrip())
            list_basefile.append(os.path.basename(line.rstrip()))
    return list_files, list_basefile


def center_print(msg: str) -> str:
    return msg.center(shutil.get_terminal_size().columns)

