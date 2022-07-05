import os
import pickle
import itertools
from constants import *
from utils import *
from create_vector_files import *


feat_separator = "$#$#$#$"
class random_pairs_genrator:

    def __init__(self,test_data_path=None):
        self.TD_test_docs = ['APW19980227.0489_TD.tml', 'APW19980227.0494_TD.tml', 'APW19980308.0201_TD.tml', 'APW19980418.0210_TD.tml',
             'CNN19980126.1600.1104_TD.tml', 'CNN19980213.2130.0155_TD.tml', 'NYT19980402.0453_TD.tml',
             'PRI19980115.2000.0186_TD.tml', 'PRI19980306.2000.1675_TD.tml']
        self.TD_dev_docs = ['APW19980227.0487_TD.tml', 'CNN19980223.1130.0960_TD.tml', 'NYT19980212.0019_TD.tml', 'PRI19980216.2000.0170_TD.tml', 'ed980111.1130.0089_TD.tml']
        self.processed_raw_data_path = os.path.join(PROCESSED_DATA_PATH,"raw_text") if test_data_path is None else test_data_path

        self.ev1_vec_extn = ".rand_ev1_vec"
        self.ev2_vec_extn = ".rand_ev2_vec"

        self.ev1_char_vec_extn = ".rand_ev1_char_vec"
        self.ev2_char_vec_extn = ".rand_ev2_char_vec"


    def get_list_of_files(self,file_extension=".events",which_files="ALL"):

        if which_files == "TEST":
            filelist = [name[:-4]+file_extension for name in self.TD_test_docs]
            te3pt_files = get_list_of_files_with_extn_in_dir(self.processed_raw_data_path, te3pt + file_extension)
            filelist = filelist + te3pt_files
        elif which_files == "DEV":
            filelist = [name[:-4] + file_extension for name in self.TD_dev_docs]
        else:
            filelist = [name[:-4] + file_extension for name in self.TD_test_docs + self.TD_dev_docs]
            te3pt_files = get_list_of_files_with_extn_in_dir(self.processed_raw_data_path, te3pt+file_extension)
            filelist = filelist+te3pt_files
        filelist.sort()

        return filelist


    def generate_event_pairs(self,filelist=None):

        if filelist is None:
            filelist = self.get_list_of_files()

        for file in filelist:
            print("####"*20)
            print("FILE : ",file)
            with open(os.path.join(self.processed_raw_data_path, file), "r") as f:
                data = f.readlines()
            # for d in data:
            #     print(d)
            # print("\n"*3)
            sent_number_list, sent_event_dict,event_context_dict,events_id_list = self.create_dicts(data)

            ev_pairs = self.gen_ev_pairs(sent_number_list,sent_event_dict,events_id_list,is_restricted_pairs=False,next_n_sent=1)
            self.write_files_with_new_event_pairs(file,event_context_dict,ev_pairs)


    def create_dicts(self,data):

        sent_event_dict = {}
        
        sent_number_list = []
        event_context_dict = {}
        events_id_list =[]
        for line in data:
            l = line.strip().split(feat_separator)
            e_id = l[0]
            e_sent_num = int(l[-1])
            e_context=feat_separator.join(l[1:])

            sent_number_list.append(e_sent_num)
            event_list = [] if sent_event_dict.get(e_sent_num,None) is None else sent_event_dict[e_sent_num]
            event_list.append(e_id)
            sent_event_dict[e_sent_num] = event_list

            event_context_dict[e_id] = e_context
            events_id_list.append(e_id)

        events_id_list = sorted(events_id_list)
        sent_number_list = sorted(list(set(sent_number_list)))
        #for debugging
        # print(sent_number_list)
        # for sent in sent_number_list:
        #     print(sent_event_dict[sent])
        return sent_number_list,sent_event_dict,event_context_dict,events_id_list

    def gen_ev_pairs(self,sent_number_list, sent_event_dict,events_id_list,is_restricted_pairs=False,next_n_sent =1):

        ev_pairs = []
        if is_restricted_pairs:
            # Form pairs only between current and next #next_n_sent sentences
            for ind,sent_number in enumerate(sent_number_list):
                event_list_for_current_sents = sent_event_dict[sent_number]
                counter =1
                # consider all events from next #next_n_sent from current sentence
                while ind+counter < len(sent_number_list) and counter<=next_n_sent:
                    # in next sentence is more than #next_n_sent away break
                    next_sent_number = sent_number_list[ind+counter]
                    if next_sent_number-sent_number > next_n_sent:
                        break
                    for ev in sent_event_dict[next_sent_number]:
                        event_list_for_current_sents.append(ev)
                    counter+=1
                # print("For sent {0} event pairs generated are as follows".format(sent_number))
                # cc = 0
                for ev_pair in itertools.combinations(event_list_for_current_sents, 2):
                    # print(ev_pair)
                    ev_pairs.append(ev_pair)
                    # cc+=1
                # print("Total combinations : ",cc)
                # print("---"*10)
        else:
            # print(events_id_list)
            # print("---" * 10)
            ev_pairs = itertools.combinations(events_id_list, 2)
            # cc=0
            # for ev_pair in ev_pairs:
            #     print(ev_pair)
            #     cc += 1
            # print("Total combinations : ", cc)

        # print(ev_pairs)
        return ev_pairs



    def write_files_with_new_event_pairs(self,file,event_context_dict,event_pairs_list):
        event1_text_file = open(os.path.join(self.processed_raw_data_path, file[:-7]+".rand_ev1"), 'w')
        event2_text_file = open(os.path.join(self.processed_raw_data_path, file[:-7] + ".rand_ev2"), 'w')

        event_pair_id_file = open(os.path.join(self.processed_raw_data_path, file[:-7] + ".rand_ev_pairs"), 'w')

        for ev1,ev2 in event_pairs_list:
            event1_text_file.write(event_context_dict[ev1]+"\n")
            event2_text_file.write(event_context_dict[ev2] + "\n")
            event_pair_id_file.write(ev1 + feat_separator + ev2 + "\n")


        event1_text_file.close()
        event2_text_file.close()
        event_pair_id_file.close()

    def generate_vector_files(self):
        self.generate_event_pairs()
        gen_vec_files = wrapper_to_initialize_class(reduced_rel_initialization)(PROCESSED_DATA_PATH)
        gen_vec_files.generate_vectors_only_for_rand_pairs()
        data_separator(VEC_FILES_PATH).segrate_docs()


    def check_if_data_for_all_test_files_generated(self):

        rand_vec_extns = [self.ev1_vec_extn,self.ev2_vec_extn,self.ev1_char_vec_extn,self.ev2_char_vec_extn]

        def _subroutine(which_files):
            all_files_present = True
            file_list = self.get_list_of_files(which_files=which_files)
            if which_files == "TEST":
                vec_data_folder_path = data_separator(VEC_FILES_PATH).test_folder_path
            else:
                vec_data_folder_path = data_separator(VEC_FILES_PATH).dev_folder_path
            for file in file_list:
                for extn in rand_vec_extns:
                    _rand_vec_file = file[:-7] + extn
                    if not os.path.exists(os.path.join(vec_data_folder_path,_rand_vec_file)):
                        print("{} file does not exist".format(_rand_vec_file))
                        all_files_present = False
            if all_files_present:
                print("All vector files for {} are present".format(which_files))

        _subroutine("TEST")
        _subroutine("DEV")




if __name__ == "__main__":

    tdg = random_pairs_genrator()
    # tdg.generate_vector_files()
    tdg.check_if_data_for_all_test_files_generated()
