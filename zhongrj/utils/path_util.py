import os
import shutil


def get_file_dir(file):
    """获取文件目录"""
    return os.path.dirname(file) + '/'


def get_project_dir():
    return get_elder(get_file_dir(__file__), 2) + '/'


def get_elder(path, n=1):
    path = os.path.abspath(path + os.path.sep + "..")
    if n == 1:
        return path
    return get_elder(path, n - 1)


def make_dir(dir):
    """创建目录"""
    if not os.path.exists(dir):
        os.makedirs(dir)


def del_dir_files(dir):
    """删除文件夹下的所有文件"""
    shutil.rmtree(dir)
    make_dir(dir)
