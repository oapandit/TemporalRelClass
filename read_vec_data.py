'''
Laods all of data for lemma training task.
'''

from __future__ import print_function
import pickle
import numpy as np
import os
from constants import *
from keras.utils.np_utils import to_categorical
import sys
from random_pairs_generator import random_pairs_genrator
from utils import *

class read_vec_data(object):
    def __init__(self,processed_data_path,num_interval_relations,num_point_relations,word_context_length,word_vector_size,char_vector_size,is_load_rand_data=False,data_set_to_load = None,is_load_only_event_head_data=False):
        self.processed_data_path = processed_data_path
        self.word_context_length = word_context_length
        self.word_vector_size = word_vector_size
        self.char_vector_size = char_vector_size
        self.num_interval_relations = num_interval_relations
        self.num_point_relations = num_point_relations
        self.num_word_for_char_emd = NUM_WORD_FOR_CHAR_EMD

        self.ev1_vec_extn = ".ev1_vec" if not is_load_rand_data else ".rand_ev1_vec"
        self.ev2_vec_extn = ".ev2_vec" if not is_load_rand_data else ".rand_ev2_vec"

        self.ev1_char_vec_extn = ".ev1_char_vec" if not is_load_rand_data else ".rand_ev1_char_vec"
        self.ev2_char_vec_extn = ".ev2_char_vec" if not is_load_rand_data else ".rand_ev2_char_vec"

        if is_load_only_event_head_data:
            self.ev1_vec_extn = self.ev1_vec_extn +event_head_vec_file_extn
            self.ev2_vec_extn = self.ev2_vec_extn +event_head_vec_file_extn

            self.ev1_char_vec_extn = self.ev1_char_vec_extn + event_head_vec_file_extn
            self.ev2_char_vec_extn = self.ev2_char_vec_extn + event_head_vec_file_extn


        self.ev1_tense_file_extn = self.ev1_vec_extn[:-4]+tense_file_extn
        self.ev2_tense_file_extn = self.ev2_vec_extn[:-4] + tense_file_extn

        self.ev1_pos_file_extn = self.ev1_vec_extn[:-4] + pos_file_extn
        self.ev2_pos_file_extn = self.ev2_vec_extn[:-4] + pos_file_extn

        self.ev1_event_num_file_extn = self.ev1_vec_extn[:-4] + event_num_file_extn
        self.ev2_event_num_file_extn = self.ev2_vec_extn[:-4] + event_num_file_extn

        self.ev1_line_num_file_extn = self.ev1_vec_extn[:-4] + line_num_file_extn
        self.ev2_line_num_file_extn = self.ev2_vec_extn[:-4] + line_num_file_extn


        self.rel_vec_extn = ".rel_vec"
        self.rel_rev_vec_extn = ".rel_rev_vec"
        self.is_load_only_event_head_data = is_load_only_event_head_data
        self.data_set_to_load = data_set_to_load
        self.is_fasttext = False

        self.w2vec_char_concate = False
        self.is_tense = False


    @property
    def load_dep_sent_vecs(self):
        return self.ev1_vec_extn

    @load_dep_sent_vecs.setter
    def load_dep_sent_vecs(self,value):
        # logger.debug("setting dep sent extns")
        self.ev1_vec_extn = ".ev1_dep_parse_words_vec"
        self.ev2_vec_extn = ".ev2_dep_parse_words_vec"
        self.word_context_length = 7


    def _delete_vague_samples(self,event_data, relation_data):
        logger.debug("\n Deleting vague entries")
        relation_data = np.argmax(relation_data, axis=1)
        vague_indices, = np.where(relation_data == 5)
        relation_data = np.delete(relation_data, vague_indices, 0)
        relation_data = to_categorical(relation_data, num_classes=5)
        logger.debug("relation data shape : {}".format(relation_data.shape))
        pruned_event_data = []
        for ind,data in enumerate(event_data):
            modified_data = np.delete(data, vague_indices, 0)
            pruned_event_data.append(modified_data)
            logger.debug("event data {0} shape : {1}".format(ind,modified_data.shape))
        return pruned_event_data,relation_data

    def _get_list_of_files(self, file_path, file_extn):
        filelist = [name for name in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, name)) and name.endswith(file_extn)]
        # logger.debug(self.data_set_to_load)
        if self.data_set_to_load is not None:
            filtered_f_list = []
            if isinstance(self.data_set_to_load,list):
                for dataset in self.data_set_to_load:
                    dataset_extn = "_"+dataset
                    for name in filelist:
                        if dataset_extn in name:
                            filtered_f_list.append(name)
            else:
                dataset_extn = "_" + self.data_set_to_load
                filtered_f_list = [name for name in filelist if dataset_extn in name]
            filelist = filtered_f_list
        filelist.sort()
        # logger.debug("\n".join(filelist))
        if len(filelist) ==0:
            logger.debug("ERROR : no file found of {0}....".format(file_extn))
            if "rand" in file_extn:
                logger.debug("Creating files with random event pairs")
                random_pairs_genrator().generate_vector_files()
                self._get_list_of_files(file_path, file_extn)
            else:
                logger.debug("exitting ......")
                sys.exit()
        return filelist


    def _load_1d_data(self,f_list):
        complete_array = None
        for fname in f_list:
            data_path = os.path.join(self.processed_data_path, fname)
            array_part = load_from_pickle_file(data_path).reshape(-1,1)
            complete_array = array_part if complete_array is None else np.concatenate((complete_array, array_part),
                                                                                      axis=0)
        return complete_array

    def _load_data_and_reshape(self,f_list,load_char_data,):
        # processed_raw_data_path = "/home/magnet/onkarp/Data/temporal_relations/processed_data/raw_text"
        complete_array = None
        for fname in f_list:
            f = open(os.path.join(self.processed_data_path, fname), 'rb')
            array_part = pickle.load(f)
            if not load_char_data:
                if self.is_load_only_event_head_data:
                    array_part = array_part.reshape(-1,self.word_vector_size)
                else:
                    array_part = array_part.reshape(-1, self.word_context_length, self.word_vector_size)
            else:
                array_part = array_part.reshape(-1,self.num_word_for_char_emd,self.char_vector_size)
                #debugging steps
                # with open(os.path.join(processed_raw_data_path, fname[:-12]+"ev_pairs"), "r") as f:
                #     raw_data_content = f.readlines()
                # if len(raw_data_content) != array_part.shape[0]:
                #     logger.debug("--"*20)
                #     logger.debug("ERROR: number of words in character embedding file : {0} is {1} but {2} in event pair files.".format(fname,array_part.shape[0],len(raw_data_content)))
                #     logger.debug("--" * 20)
                # else:
                #     logger.debug("This file {0} is right in dimension".format(fname))
            complete_array = array_part if complete_array is None else np.concatenate((complete_array, array_part), axis=0)
        return complete_array


    def _read_relations(self,rel_vec_extn,num_relations):
        # rel_vec_extn = ".rel_vec"
        f_list = self._get_list_of_files(self.processed_data_path, rel_vec_extn)
        logger.debug(len(f_list))
        # logger.debug("List of files : ",f_list)
        complete_array = None
        for fname in f_list:
            f = open(os.path.join(self.processed_data_path, fname), 'rb')
            array_part = pickle.load(f)
            array_part = array_part.reshape(-1,num_relations)
            complete_array = array_part if complete_array is None else np.concatenate((complete_array, array_part),
                                                                                      axis=0)
            f.close()
        return complete_array


    def load_events_word(self, load_reverse_data,load_char_data=False):
        ev_vec_extn = self.ev1_vec_extn if not load_char_data else self.ev1_char_vec_extn
        logger.debug(ev_vec_extn)
        f_list = self._get_list_of_files(self.processed_data_path, ev_vec_extn)
        logger.debug(len(f_list))
        event1_data = self._load_data_and_reshape(f_list,load_char_data)

        ev_vec_extn = self.ev2_vec_extn if not load_char_data else self.ev2_char_vec_extn
        logger.debug(ev_vec_extn)
        f_list = self._get_list_of_files(self.processed_data_path, ev_vec_extn)
        logger.debug(len(f_list))
        event2_data = self._load_data_and_reshape(f_list,load_char_data)

        if load_reverse_data:
            logger.debug("Returning reverse concatenated data")
            ev1_data = np.concatenate((event1_data, event2_data), axis=0)
            ev2_data = np.concatenate((event2_data, event1_data), axis=0)
            return ev1_data,ev2_data

        return event1_data,event2_data



    def load_feat_data(self,ev1_file_extn,ev2_file_extn,load_reverse_data):
        f_list = self._get_list_of_files(self.processed_data_path,ev1_file_extn)
        logger.debug(len(f_list))
        ev1_feat_data = self._load_1d_data(f_list)

        f_list = self._get_list_of_files(self.processed_data_path,ev2_file_extn)
        logger.debug(len(f_list))
        ev2_feat_data = self._load_1d_data(f_list)

        if load_reverse_data:
            logger.debug("Returning reverse concatenated data")
            ev1_data = np.concatenate((ev1_feat_data, ev2_feat_data), axis=0)
            ev2_data = np.concatenate((ev2_feat_data, ev1_feat_data), axis=0)
            return ev1_data,ev2_data

        return ev1_feat_data,ev2_feat_data

    def load_interval_relations(self, load_reverse_data):
        if load_reverse_data:
            rel_vec_data = self._read_relations(rel_vec_extn = self.rel_vec_extn,num_relations=self.num_interval_relations)
            rel_rev_vec_data = self._read_relations(rel_vec_extn=self.rel_rev_vec_extn,num_relations=self.num_interval_relations)
            logger.debug("Returning reverse concatenated data")
            return np.concatenate((rel_vec_data, rel_rev_vec_data), axis=0)

        else:
            return self._read_relations(rel_vec_extn = ".rel_vec",num_relations=self.num_interval_relations)


    def load_point_relations(self, load_reverse_data):

        end_point_vec_file_extn = [".end_point1",".end_point2",".end_point3",".end_point4"]
        end_point_rev_vec_file_extn = [".rev_end_point1", ".rev_end_point2", ".rev_end_point3", ".rev_end_point4"]

        end_point_vecs= [0]*4

        for i in range(len(end_point_vecs)):
            end_point_vecs[i] = self._read_relations(rel_vec_extn=end_point_vec_file_extn[i], num_relations=self.num_point_relations)
            if load_reverse_data:
                end_point_rev_vec = self._read_relations(rel_vec_extn=end_point_rev_vec_file_extn[i],
                                                         num_relations=self.num_point_relations)
                end_point_vecs[i] = np.concatenate((end_point_vecs[i], end_point_rev_vec), axis=0)
        return end_point_vecs


    def data_generator_temp_relations(self,load_reverse_data,load_char_data):

        # ev1_vec_extn = ".ev1_vec"
        # ev2_vec_extn = ".ev2_vec"
        # rel_vec_extn = ".rel_vec"

        f_list = self.get_file_name_list()

        while 1:
            for f in f_list:
                # logger.debug(f)
                f_substring = f[:-4]

                ev1_data_file = f_substring + self.ev1_vec_extn
                ev2_data_file = f_substring + self.ev2_vec_extn

                ev1_char_data_file = f_substring + self.ev1_char_vec_extn
                ev2_char_data_file = f_substring + self.ev2_char_vec_extn

                rel_data_file = f_substring + self.rel_vec_extn
                rev_rel_data_file = f_substring + self.rel_rev_vec_extn

                rel_data = load_from_pickle_file(os.path.join(self.processed_data_path,rel_data_file)).reshape(-1, self.num_interval_relations)
                event1_data = load_from_pickle_file(os.path.join(self.processed_data_path,ev1_data_file)).reshape(-1,self.word_context_length,self.word_vector_size)
                event2_data = load_from_pickle_file(os.path.join(self.processed_data_path,ev2_data_file)).reshape(-1,self.word_context_length,self.word_vector_size)

                if load_char_data:
                    event1_char_data = load_from_pickle_file(os.path.join(self.processed_data_path,ev1_char_data_file)).reshape(-1,self.num_word_for_char_emd,self.char_vector_size)
                    event2_char_data = load_from_pickle_file(
                        os.path.join(self.processed_data_path, ev2_char_data_file)).reshape(-1,
                                                                                            self.num_word_for_char_emd,
                                                                                            self.char_vector_size)
                if load_reverse_data:

                    rev_rel_data = load_from_pickle_file(os.path.join(self.processed_data_path,rev_rel_data_file)).reshape(-1, self.num_interval_relations)
                    rel_data = np.concatenate((rel_data, rev_rel_data), axis=0)

                    ev1_data = np.concatenate((event1_data, event2_data), axis=0)
                    ev2_data = np.concatenate((event2_data, event1_data), axis=0)

                    event1_data = ev1_data
                    event2_data = ev2_data

                    if load_char_data:
                        ev1_data = np.concatenate((event1_char_data, event2_char_data), axis=0)
                        ev2_data = np.concatenate((event2_char_data, event1_char_data), axis=0)

                        event1_char_data = ev1_data
                        event2_char_data = ev2_data

                        # logger.debug("event1_char_data" + " : shape : " + str(event1_char_data.shape))
                        # logger.debug("event2_char_data" + " : shape : " + str(event2_char_data.shape))

                # logger.debug("event1_word_data" + " : shape : " + str(event1_data.shape))
                # logger.debug("event2_word_data" + " : shape : " + str(event2_data.shape))


                # logger.debug("Interval relation data" + " : shape : " + str(rel_data.shape))
                if load_char_data:
                    event_data = [event1_data, event1_char_data, event2_data, event2_char_data]
                else:
                    event_data = [event1_data, event2_data]

                yield event_data,rel_data

    def get_file_name_list(self):
        f_list = self._get_list_of_files(self.processed_data_path, self.ev1_vec_extn)
        f_list = [f[0:f.rfind('.')]+".tml" for f in f_list]
        return f_list


    def load_data(self,load_reverse_data,load_point_relations,load_char_data,is_train_without_vague,load_pos_data):

        # self.ev1_vec_extn = ".ev1_fasttext_char_all_cont_vec"
        # self.ev2_vec_extn = ".ev2_fasttext_char_all_cont_vec"

        event1_data,event2_data= self.load_events_word(load_reverse_data)
        logger.debug("event1_word_data" + " : shape : " + str(event1_data.shape))
        logger.debug("event2_word_data" + " : shape : " + str(event2_data.shape))

        if load_char_data:
            event1_char_data, event2_char_data = self.load_events_word(load_reverse_data,load_char_data)
            logger.debug("event1_char_data" + " : shape : " + str(event1_char_data.shape))
            logger.debug("event2_char_data" + " : shape : " + str(event2_char_data.shape))

        if load_point_relations:
            relation_data = self.load_point_relations(load_reverse_data)
            for i in range(len(relation_data)):
                logger.debug("Point relation data " + str(i) + " : shape : " + str(relation_data[i].shape))
        else:
            relation_data = self.load_interval_relations(load_reverse_data)
            logger.debug("Interval relation data" + " : shape : " + str(relation_data.shape))

        if load_pos_data:
            logger.debug(self.ev1_pos_file_extn)
            logger.debug(self.ev2_pos_file_extn)
            ev1_pos_data, ev2_pos_data = self.load_feat_data(self.ev1_pos_file_extn,self.ev2_pos_file_extn,load_reverse_data)
            logger.debug("event1_pos_data" + " : shape : " + str(ev1_pos_data.shape))
            logger.debug("event2_pos_data" + " : shape : " + str(ev2_pos_data.shape))


        if self.is_fasttext:
            self.ev1_char_vec_extn = ".ev1_fasttext_char_vec"
            self.ev2_char_vec_extn = ".ev2_fasttext_char_vec"

            self.num_word_for_char_emd, self.char_vector_size = 1,300
            load_char_data = True
            event1_char_data, event2_char_data = self.load_events_word(load_reverse_data, load_char_data)
            logger.debug("fasttext event1_char_data" + " : shape : " + str(event1_char_data.shape))
            logger.debug("fasttext event2_char_data" + " : shape : " + str(event2_char_data.shape))


        if self.w2vec_char_concate:

            self.ev1_char_vec_extn = ".ev1_fasttext_char_all_cont_vec"
            self.ev2_char_vec_extn = ".ev2_fasttext_char_all_cont_vec"

            event1_char_all_cont_data, event2_char_all_cont_data = self.load_events_word(load_reverse_data)
            logger.debug("event1_char_all_cont_data" + " : shape : " + str(event1_char_all_cont_data.shape))
            logger.debug("event2_char_all_cont_data" + " : shape : " + str(event2_char_all_cont_data.shape))

            event1_data = np.concatenate((event1_data, event1_char_all_cont_data), axis=2)
            event2_data = np.concatenate((event2_data, event2_char_all_cont_data), axis=2)


        if self.is_tense:
            logger.debug(self.ev1_tense_file_extn)
            logger.debug(self.ev2_tense_file_extn)
            ev1_tense_data, ev2_tense_data = self.load_feat_data(self.ev1_tense_file_extn,self.ev2_tense_file_extn,load_reverse_data)
            logger.debug("ev1_tense_data" + " : shape : " + str(ev1_tense_data.shape))
            logger.debug("ev2_tense_data" + " : shape : " + str(ev2_tense_data.shape))


        if load_pos_data:
            if load_char_data:
                event_data = [event1_data, event1_char_data,ev1_pos_data, event2_data, event2_char_data,ev2_pos_data]
            else:
                event_data = [event1_data,ev1_pos_data, event2_data,ev2_pos_data]
        else:
            if load_char_data:
                event_data = [event1_data,event1_char_data,event2_data,event2_char_data]
            else:
                event_data = [event1_data,event2_data]

        if is_train_without_vague:
            event_data, relation_data = self._delete_vague_samples(event_data, relation_data)

        return event_data,relation_data



if __name__ == "__main__":
    pass
