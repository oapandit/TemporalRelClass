# from __future__ import print_function
import pickle
import numpy as np
import os
from constants import *
from utils import *
from apetite.TimeMLDocument import Document
import time

class CorpusStats:

    def __init__(self,processed_data_path):
        self.processed_data_path = processed_data_path
        self.raw_text_path = os.path.join(processed_data_path, "raw_text")
        self.tml_files_dir = TML_FILES_PATH
        self.tml_file_extn = "*.tml"


    def get_list_of_files(self,dir_path,file_extension=".rel"):

        filelist = [name for name in os.listdir(dir_path) if name.endswith(file_extension) and not os.path.isdir(os.path.join(dir_path, name))]
        filelist.sort()
        return filelist


    def get_class_stats(self,data_set=None):

        rel_count_dict = {}

        #Read all relation files if specific dataset is not mentioned
        if data_set is None:
            files = self.get_list_of_files(self.raw_text_path)
        if data_set == "TE3PT":
            file_extn = "_TE3PT.rel"
            files = self.get_list_of_files(self.raw_text_path,file_extension=file_extn)
        for file in files:

            print "File : ", file
            print "\n \n"
            with open(os.path.join(self.raw_text_path, file), "r") as f:
                raw_data_content = f.readlines()

            for sent in raw_data_content:
                word = sent.strip().upper()
                rel_count_dict[word] = rel_count_dict.get(word,0)+1


        print "Number of relations"
        for key, value in sorted(rel_count_dict.iteritems(), key=lambda (k, v): (v, k)):
            print key, value
        # for k, v in rel_count_dict.iteritems():
        #     print
        #     k, v


    def check_if_consistent(self):
        tml_file_list = search_all_file_in_dir(self.tml_files_dir,self.tml_file_extn)

        num_tml_files = len(tml_file_list)
        num_inconsistency = 0
        # tml_file_list= ["/home/magnet/onkarp/Data/temporal_relations/raw_data/td_dataset/PRI19980121.2000.2591_TD.tml"]
        for tml_file in tml_file_list:
            print tml_file
            temp_graph = Document(tml_file).get_graph()
            temp_graph.index_on_node1()
            # t0 = time.time()
            is_consistent = temp_graph.saturate()
            # t1 = time.time()
            # print "saturation of graph in {0}s".format((t1 - t0))
            if not is_consistent:
                print "File : {0} is inconsistent.".format(tml_file)
                num_inconsistency+=1

        print "Inconsistent tml files : {0} out of {1}...{2}% inconsistency".format(num_inconsistency,num_tml_files,num_inconsistency*100/float(num_tml_files))






if __name__ == "__main__":


    processed_data_path = "/home/magnet/onkarp/Data/temporal_relations/processed_data"

    c = CorpusStats(processed_data_path)
    c.get_class_stats(data_set="TE3PT")
    # c.check_if_consistent()