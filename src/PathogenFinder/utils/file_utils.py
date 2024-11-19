import os


def get_filename(file_path):
    filebase = os.path.basename(file_path)
    ext = os.path.splitext(filebase)[1:]
    filename = os.path.splitext(filebase)[0]
    return filename, ext
    

