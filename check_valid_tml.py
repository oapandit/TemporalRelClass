import os
import pickle
from utils import *
from constants import *
import TE3_evaluation as te
from subprocess import Popen, PIPE
import shlex
import re

class CheckValidTML:

    def __init__(self,tml_path):
        self.tml_path = tml_path
        self.tml_file_pattern = "*.tml"
        self.tml_validator_jar = "/home/magnet/onkarp/Code/temporal_relations/TML_validator/TimeML-validator.jar"



    def check_all_tml_files(self):

        tml_files = search_all_file_in_dir(self.tml_path,self.tml_file_pattern)

        total_tml_files = len(tml_files)
        print("There are total {0} tml files.".format(total_tml_files))

        error_count =0
        err_file_list = []

        # tml_files = ['/home/magnet/onkarp/Data/temporal_relations/raw_data/timebank_1_2/data/extra/wsj_0329.tml']

        for tml_file in tml_files:
            print(tml_file)
            command = 'java -jar ' + self.tml_validator_jar + ' -d ' + tml_file
            # os.system(command)
            process = Popen(shlex.split(command), stdout=PIPE, stderr=PIPE)
            out = process.communicate()

            m = re.findall(r'error|exception', out[0], re.I)

            if m is not None:
                error_count+=1
                err_file_list.append(tml_file)
                print "ERROR in file "+tml_file
                print("----"*10)
                # print(out[0])
                # for o in out:
                    # pass
                    # print(o)
            # output = out[stdout]
        print("Error found in {0} in files".format(error_count))

if __name__ == '__main__':
    path = "/home/magnet/onkarp/Data/temporal_relations/raw_data/"
    cv = CheckValidTML(path)
    cv.check_all_tml_files()
