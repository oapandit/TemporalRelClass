import os
import pickle
from constants import *
from utils import *


class tests(object):
    def __init__(self):
        self.tml_files_path = raw_data_path
        self.raw_text_path = processed_text_data_path
        self.vec_file_path = processed_vec_data_path
        self.ev1_extn = ".ev1"
        self.ev2_extn = ".ev2"
        self.event_file_extn = ".events"
        self.event_pairs_file_extn = ".ev_pairs"
        self.rel_file_extn = ".rel"

        self.ev1_vec_extn = ".ev1_vec"
        self.ev2_vec_extn = ".ev2_vec"
        self.rel_vec_extn = ".rel_vec"
        self.rel_rev_vec_extn = ".rel_rev_vec"
        self.ev1_dep_vec_extn = ".ev1_dep_parse_words_vec"
        self.ev2_dep_vec_extn = ".ev2_dep_parse_words_vec"


    def _get_list_of_event_files(self,file_path,file_extn):
        file_list = get_list_of_files_in_dir(file_path)
        file_list = [name for name in file_list if name.endswith(file_extn)]
        file_list.sort()
        return file_list


    def test_is_every_file_present(self):
        no_error = True
        events_file_list = self._get_list_of_event_files(self.raw_text_path,self.event_file_extn)
        ev1_file_list = self._get_list_of_event_files(self.raw_text_path,self.ev1_extn)
        ev2_file_list = self._get_list_of_event_files(self.raw_text_path, self.ev2_extn)
        rel_file_list = self._get_list_of_event_files(self.raw_text_path, self.rel_file_extn)
        event_pairs_file_list = self._get_list_of_event_files(self.raw_text_path,self.event_pairs_file_extn)

        if not (len(events_file_list) == len(ev1_file_list) == len(ev2_file_list) == len(rel_file_list) == len(event_pairs_file_list)):
            logger.debug("All files do not exist, some descripency!")
            no_error = False

        logger.debug("Number of event files :{}".format(len(events_file_list)))
        logger.debug("Number of ev1 files :{}".format(len(ev1_file_list)))
        logger.debug("Number of ev2 files :{}".format(len(ev2_file_list)))
        logger.debug("Number of rel files :{}".format(len(rel_file_list)))
        logger.debug("Number of event pairs files :{}".format(len(event_pairs_file_list)))


        if no_error:
            logger.debug("All is well; equal number of files in raw text")

    def test_num_events_num_relations_in_raw(self):

        no_error = True

        events_file_list = self._get_list_of_event_files(self.raw_text_path,self.event_file_extn)
        ev1_file_list = self._get_list_of_event_files(self.raw_text_path,self.ev1_extn)
        ev2_file_list = self._get_list_of_event_files(self.raw_text_path, self.ev2_extn)
        rel_file_list = self._get_list_of_event_files(self.raw_text_path, self.rel_file_extn)
        event_pairs_file_list = self._get_list_of_event_files(self.raw_text_path,self.event_pairs_file_extn)

        _ev1_f_list = []
        _ev2_f_list = []
        _rel_f_list = []

        for file in event_pairs_file_list:
            ev1_fname = file[:-9]+self.ev1_extn
            ev2_fname = file[:-9]+self.ev2_extn
            rel_fname = file[:-9] + self.rel_file_extn

            _ev1_f_list.append(ev1_fname)
            _ev2_f_list.append(ev2_fname)
            _rel_f_list.append(rel_fname)

            with open(os.path.join(self.raw_text_path,file),'r') as f:
                ev_pairs_data = f.readlines()

            with open(os.path.join(self.raw_text_path,ev1_fname),'r') as f:
                ev1_data = f.readlines()

            with open(os.path.join(self.raw_text_path,ev2_fname),'r') as f:
                ev2_data = f.readlines()

            with open(os.path.join(self.raw_text_path,rel_fname),'r') as f:
                rel_data = f.readlines()


            if not(len(ev_pairs_data) == len(ev1_data) == len(ev2_data) == len(rel_data)):
                logger.debug(20*"^^^^")
                logger.debug("File : {}".format(file))
                logger.debug("Something is wrong; Not equal number of events pairs or relations")
                logger.debug("Number of ev1 :{}".format(len(ev1_data)))
                logger.debug("Number of ev2 :{}".format( len(ev2_data)))
                logger.debug("Number of rel :{}".format(len(rel_data)))
                logger.debug("Number of event pairs :{}".format( len(ev_pairs_data)))

                no_error = False

        _ev1_f_list.sort()
        _ev2_f_list.sort()
        _rel_f_list.sort()

        if ev1_file_list != _ev1_f_list:
            logger.debug("ev1 files are not same in number")
            no_error = False

        if ev2_file_list != _ev2_f_list:
            logger.debug("ev2 files are not same in number")
            no_error = False

        if rel_file_list != _rel_f_list:
            logger.debug("rel files are not same in number")
            no_error = False


        if no_error:
            logger.debug("All is well; files have equal number of events and relations")



    def test_ev_rel_files_in_vec(self):

        dirs = get_list_of_dirs(self.vec_file_path)
        dirs = [""] if len(dirs)==0 else dirs

        for dir in dirs:
            logger.debug(20*"$$$$")
            logger.debug(dir)
            logger.debug(20 * "$$$$")
            no_error = True
            ev1_file_list = self._get_list_of_event_files(os.path.join(self.vec_file_path, dir), self.ev1_vec_extn)
            ev2_file_list = self._get_list_of_event_files(os.path.join(self.vec_file_path, dir), self.ev2_vec_extn)
            rel_file_list = self._get_list_of_event_files(os.path.join(self.vec_file_path, dir), self.rel_vec_extn)
            ev1_dep_file_list = self._get_list_of_event_files(os.path.join(self.vec_file_path, dir), self.ev1_dep_vec_extn)
            ev2_dep_file_list = self._get_list_of_event_files(os.path.join(self.vec_file_path, dir), self.ev2_dep_vec_extn)
            rel_rev_file_list = self._get_list_of_event_files(os.path.join(self.vec_file_path, dir), self.rel_rev_vec_extn)


            if not (len(ev1_file_list) == len(ev2_file_list) == len(rel_file_list) == len(ev1_dep_file_list) == len(ev2_dep_file_list) == len(rel_rev_file_list)):
                logger.debug(20*"^^^^^")
                logger.debug("All vector files do not exist, some descripency!")
                no_error = False

                for file in ev1_dep_file_list:
                    if file[:-24]+self.ev1_vec_extn not in ev1_file_list:
                        logger.debug(file)

            logger.debug("Number of ev1 files :{}".format(len(ev1_file_list)))
            logger.debug("Number of ev2 files :{}".format(len(ev2_file_list)))
            logger.debug("Number of rel files :{}".format(len(rel_file_list)))
            logger.debug("Number of dep ev1 files :{}".format(len(ev1_dep_file_list)))
            logger.debug("Number of dep ev2 files :{}".format(len(ev2_dep_file_list)))
            logger.debug("Number of reverse relation files :{}".format(len(rel_rev_file_list)))



            if no_error:
                logger.debug("All is well; vector files have equal number of events and relations")

    def test_count_ev_rel_in_vec_files(self):

        dirs = get_list_of_dirs(self.vec_file_path)
        dirs = [""] if len(dirs)==0 else dirs

        for dir in dirs:
            logger.debug(20*"$$$$")
            logger.debug(dir)
            logger.debug(20 * "$$$$")
            no_error = True

            _data_path = os.path.join(self.vec_file_path, dir)
            ev1_file_list = self._get_list_of_event_files(_data_path, self.ev1_vec_extn)

            for file in ev1_file_list:
                is_dep_vec_created = True
                with open(os.path.join(self.raw_text_path, file[0:file.rfind(".")]+self.event_pairs_file_extn), 'r') as f:
                    ev_pairs_data = f.readlines()
                ev1_vec = load_from_pickle_file(os.path.join(_data_path,file)).reshape(-1,9,300)
                ev2_vec = load_from_pickle_file(os.path.join(_data_path, file[:-8]+self.ev2_vec_extn)).reshape(-1, 9, 300)

                if os.path.exists(os.path.join(_data_path,file[:-8]+self.ev1_dep_vec_extn)):
                    ev1_dep_vec = load_from_pickle_file(os.path.join(_data_path,file[:-8]+self.ev1_dep_vec_extn)).reshape(-1,7,300)
                    ev2_dep_vec = load_from_pickle_file(os.path.join(_data_path, file[:-8]+self.ev2_dep_vec_extn)).reshape(-1, 7, 300)
                else:
                    is_dep_vec_created = False
                rel_vec = load_from_pickle_file(
                    os.path.join(_data_path, file[:-8] + self.rel_vec_extn)).reshape(-1,6)
                rel_rev_vec = load_from_pickle_file(
                    os.path.join(_data_path, file[:-8] + self.rel_rev_vec_extn)).reshape(-1,6)

                logger.debug(20*"^^^^^")
                logger.debug("File {} ".format(file))

                logger.debug("Number of ev1 files {} ".format(ev1_vec.shape[0]))
                logger.debug("Number of ev2 files {} ".format(ev2_vec.shape[0]))
                logger.debug("Number of rel files {} ".format(rel_vec.shape[0]))
                logger.debug("Number of reverse relation files {} ".format(rel_rev_vec.shape[0]))
                logger.debug("Number of event pairs {} ".format(len(ev_pairs_data)))

                if is_dep_vec_created:
                    logger.debug("Number of dep ev1 files {} ".format(ev1_dep_vec.shape[0]))
                    logger.debug("Number of dep ev2 files {} ".format(ev2_dep_vec.shape[0]))


                    if not (ev1_vec.shape[0] == ev2_vec.shape[0] == ev1_dep_vec.shape[0] == ev2_dep_vec.shape[0] == rel_vec.shape[0] == rel_rev_vec.shape[0] == len(ev_pairs_data)):
                        no_error = False
                        logger.debug("All vector are not of same length, some descripency!")

                else:
                    if not (ev1_vec.shape[0] == ev2_vec.shape[0] ==
                            rel_vec.shape[0] == rel_rev_vec.shape[0] == len(ev_pairs_data)):
                        no_error = False
                        logger.debug("All vector are not of same length, some descripency!")




            if no_error:
                logger.debug("All is well; vector files have equal number of events and relations")


    def test_event_head_word_is_not_word(self):
        events_file_list = self._get_list_of_event_files(self.raw_text_path, self.event_file_extn)
        for events_file in events_file_list:
            # logger.debug(20*"====")

            with open(os.path.join(self.raw_text_path, events_file), 'r') as f:
                ev_file_data = f.readlines()

            for line in ev_file_data:
                ev_head_word = line.split(feat_separator)[5]
                context = line.split(feat_separator)[6]
                words = context.split()
                # if words[4]!= ev_head_word:
                #     logger.debug(20 * "==")
                #     logger.debug(events_file)
                #     logger.debug("ev head word {} and from context {}".format(ev_head_word,words[4]))
                if not ev_head_word.isalpha():
                    logger.debug(20 * "==")
                    logger.debug(events_file)
                    logger.debug("ev head word {} ".format(ev_head_word))






if __name__ == '__main__':

    t = tests()
    # t.test_is_every_file_present()
    # t.test_num_events_num_relations_in_raw()
    t.test_ev_rel_files_in_vec()
    t.test_count_ev_rel_in_vec_files()
    t.test_event_head_word_is_not_word()