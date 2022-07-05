import pickle
from constants import *
import os
import fnmatch
import sys
import signal
import shutil
import logging
from logging.handlers import RotatingFileHandler
import time

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Logger settings -------------------------------------------------

logger = logging.getLogger()
timestr = time.strftime("%d%m%Y-%H%M%S")
logger.setLevel(logging.DEBUG)
create_dir(log_files_path)

if is_log_debug_messages:
    log_format = ('[%(levelname)-8s %(filename)s:%(lineno)s] %(message)s')
    # output debug logs to this file
    debug_file = os.path.join(log_files_path,timestr+'.debug.log')
    # fh = logging.FileHandler(debug_file)
    fh = RotatingFileHandler(debug_file, maxBytes=10000000, backupCount=2)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# output only info logs to this file
log_format = ('%(message)s')
info_file = os.path.join(log_files_path,timestr+'.info.log')
# fh = logging.FileHandler(info_file)
fh = RotatingFileHandler(info_file, maxBytes=10000000, backupCount=2)
fh.setLevel(logging.INFO)
formatter = logging.Formatter(log_format)
fh.setFormatter(formatter)
logger.addHandler(fh)

#-------------------------------------------------------------------

def get_list_of_files_with_extn_in_dir(file_path, file_extn):
    filelist = [name for name in os.listdir(file_path) if
                os.path.isfile(os.path.join(file_path, name)) and name.endswith(file_extn)]
    filelist.sort()
    return filelist

def get_list_of_files_in_dir(file_path):
    filelist = [name for name in os.listdir(file_path) if
                os.path.isfile(os.path.join(file_path, name))]
    filelist.sort()
    return filelist

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_as_pickle_file(vector_data,f_path):
    f = open(f_path, "wb")
    pickle.dump(vector_data, f)
    f.close()

def load_from_pickle_file(f_path):
    f = open(f_path, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def search_single_file_in_dir(path, f_name):

    is_file_found = False
    for root, dirs, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename, f_name):
                filename = os.path.join(root, basename)
                is_file_found = True
                return filename

    if not is_file_found:
        print("ERROR : {0} does not exist in directory {1}".format(f_name,path))
        return None



class TimedOutExc(Exception):
    pass

def deadline(timeout, *args):
    def decorate(f):
        def handler(signum, frame):
            raise TimedOutExc()

        def new_f(*args):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)
            return f(*args)
            signal.alarm(0)

        new_f.__name__ = f.__name__
        return new_f
    return decorate

def search_all_file_in_dir(path,pattern):

    files_list = []
    is_file_found = False
    for root, dirs, files in os.walk(path):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                files_list.append(os.path.join(root, basename))
                is_file_found = True


    if not is_file_found:
        print("ERROR : No file with {0} pattern in directory {1}".format(pattern,path));
        sys.exit();

    return files_list


def delete_all_data_from_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def get_last_lid(tlinks):
    lid = []
    for tlink in tlinks:
        lid.append(int(tlink.attrib["lid"][1:]))
    return max(lid)

def get_list_of_dirs(path):
    return [name for name in os.listdir(path) if os.path.isdir(name)]