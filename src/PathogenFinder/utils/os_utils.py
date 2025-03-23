import os
import datetime


def get_filename(file_path:str) -> (str,str):
    filebase = os.path.basename(file_path)
    ext = os.path.splitext(filebase)[1:]
    filename = os.path.splitext(filebase)[0]
    return filename, ext
    

def create_outputfolder(outpath:str) -> str:
    out_folder = '{outpath}_{date:%Y-%m-%d_%H-%M-%S}'.format(outpath=outpath,
                                                               date=datetime.datetime.now())
    out_paths = {"main": out_folder,
                 "conf": "{}/configuration".format(out_folder)
                 }

    for k, v in out_paths.items():
        os.mkdir(v)

    return out_paths


