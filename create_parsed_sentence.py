import os
from nltk.tokenize import sent_tokenize
import nltk
from constants import *
from utils import *


from nltk.parse.stanford import StanfordDependencyParser

path_to_jar = '/home/magnet/onkarp/nltk_data/stanford-parser-full-2018-10-17/stanford-parser.jar'
path_to_models_jar = '/home/magnet/onkarp/nltk_data/stanford-english-corenlp-2018-10-05-models.jar'



class CreateParsedSentence:

    def __init__(self,raw_text_path):
        self.dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)
        self.raw_text_path = raw_text_path
        self.parse_sen_file_extn = "_dep_parse_words"
        self.ev1_extn = ".ev1"
        self.ev2_extn = ".ev2"
        self.event_file_extn = ".events"
        self.event_pairs_file_extn = ".ev_pairs"
        self.len_of_dep_context = []


    def _get_list_of_ev1_ev2_files(self, file_path):

        file_list = get_list_of_files_in_dir(file_path)
        file_list = [name for name in file_list if name.endswith(self.ev1_extn) or name.endswith(self.ev2_extn)]
        file_list.sort()
        return file_list

    def _get_list_of_event_files(self,file_path):
        file_list = get_list_of_files_in_dir(file_path)
        file_list = [name for name in file_list if name.endswith(self.event_file_extn)]
        file_list.sort()
        return file_list

    def _get_position_of_event_in_tree(self,tree,word):

        num_nodes = len(tree.nodes) +5

        event_position = [] # if in sentence more than one occurance of event word.
        # for i in range(1,num_nodes):
        #     if word == tree.nodes[i]['word']:
        #         event_position.append(i)
        i =1
        while word != tree.nodes[i]['word']:
            i+=1
            if i >500:
                print(20 * "+++++")
                print "ERROR"
                # print "Too much; unable to find word in tree"
                print "word is ",word
                # sys.exit()
                # return None
                raise Exception('"Too much; unable to find word in tree"')
        event_position.append(i)

        # print "word is {} and position is {}".format(word,event_position[0])

        # if len(event_position)==0:
        #     print "ERROR"
        #     print "Event word not in tree."
        #     print(tree.tree())
        #     raise Exception('"Too much; unable to find word in tree"')
        #
        # if len(event_position) >1:
        #     print("More than one event words in sentence")
        #     print("Returning first")


        return event_position[0]





    def _get_path_to_root_for_word(self,tree,word_position):

        words_in_path = []
        # word_position +=1 # in nltk tree; 0 is reserved for dummy root node; and actual words start from 1
        current_word_postion = word_position
        words_in_path.append(tree.nodes[current_word_postion]['word']) # adding event head word first
        while tree.nodes[current_word_postion]['head'] !=0:
            current_word_postion = tree.nodes[current_word_postion]['head']
            words_in_path.append(tree.nodes[current_word_postion]['word'])

        self.len_of_dep_context.append(len(words_in_path))
        context_from_dep_path = " ".join(words_in_path)
        return context_from_dep_path

    def _get_dependancy_tree(self,sentence):
        try:
            result = self.dependency_parser.raw_parse(sentence)
            dep = result.next()
        except:
            raise Exception('Error in getting dependency tree.')
        return dep

    def remove_multiple_sentences(self,sentence,event_word):
        sentences = sent_tokenize(sentence)
        for sentence in sentences:
            if event_word in sentence:
                return sentence


    def get_path_for_sentence_and_event(self,sentence,event_word):

        try:

            sentence = self.remove_multiple_sentences(sentence,event_word)

            # print "Sentence is ",sentence

            tree = self._get_dependancy_tree(sentence)

            # print "Tree is ",tree.tree()
            # getting event word position from sentence itself
            event_word_pos_in_sent = self._get_position_of_event_in_tree(tree, event_word)
            context_from_dep_path = self._get_path_to_root_for_word(tree, event_word_pos_in_sent)
            return context_from_dep_path
        except:
            print "Error : somehow unable to find context from tree. Returning default word "
            return PAD_WORD


    def generate_dep_sentences_file(self,fname):
        with open(os.path.join(self.raw_text_path, fname), "r") as f:
            raw_data = f.readlines()

        parsed_dep_sent_path = os.path.join(self.raw_text_path, fname+self.parse_sen_file_extn)

        if not os.path.exists(parsed_dep_sent_path):
            dep_path_context_file = open(parsed_dep_sent_path, 'w')
            for data in raw_data:

                # print("raw data ",data)

                _data = data.strip().split(feat_separator)

                event_word = _data[4]
                sentence = _data[8]
                event_word_pos_in_sent = int(_data[9])-1

                context_from_dep_path = self.get_path_for_sentence_and_event(sentence,event_word)
                print(context_from_dep_path)
                print(20*"+++++")
                dep_path_context_file.write(context_from_dep_path + "\n")

            dep_path_context_file.close()

        else:
            print "file exists ..."



    def generate_for_all_files_directly(self):

        file_list = self._get_list_of_ev1_ev2_files(self.raw_text_path)

        for f in file_list:
            print "\n File : ",f
            self.generate_dep_sentences_file(f)
            print 20*"===="

        fdist = nltk.FreqDist(self.len_of_dep_context)

        # Output top 50 words

        for word, frequency in fdist.most_common(50):
            print(u'{};{}'.format(word, frequency))


    def generate_for_all_files_from_events(self,file_list=None):
        if file_list is None:
            file_list = self._get_list_of_event_files(self.raw_text_path)
        for file in file_list:
            # if not "APW19980227.0489" in file:
            print 20 * "===="
            print "\n File : ",file
            dep_ev1_file = os.path.join(self.raw_text_path, file[:-7]+".ev1" + self.parse_sen_file_extn)
            dep_ev2_file = os.path.join(self.raw_text_path, file[:-7]+".ev2" + self.parse_sen_file_extn)

            if not os.path.exists(dep_ev1_file) or not os.path.exists(dep_ev2_file):
                is_ev1_file_exists = False
                is_ev2_file_exists = False
                with open(os.path.join(self.raw_text_path,file), "r") as f:
                    raw_data = f.readlines()
                event_sent_dict = {}
                for data in raw_data:
                    event_info = data.strip().split(feat_separator)
                    eiid = event_info[0]
                    event_word = event_info[5]
                    sent = event_info[9]

                    print sent
                    print "word : ",event_word

                    context_from_dep_path = self.get_path_for_sentence_and_event(sent,event_word)
                    print(20 * "+++++")
                    event_sent_dict[eiid] = context_from_dep_path

                ev_pairs_file = file[:-7]+self.event_pairs_file_extn
                with open(os.path.join(self.raw_text_path,ev_pairs_file), "r") as f:
                    raw_data = f.readlines()


                if not os.path.exists(dep_ev1_file):
                    dep_ev1_file = open(dep_ev1_file, 'w')
                else:
                    is_ev1_file_exists = True

                if not os.path.exists(dep_ev2_file):
                    dep_ev2_file = open(dep_ev2_file,'w')
                else:
                    is_ev2_file_exists = True

                for data in raw_data:
                    pair_info =data.strip().split(feat_separator)
                    eiid_ev1 = pair_info[0]
                    eiid_ev2 = pair_info[1]
                    if not is_ev1_file_exists:
                        dep_ev1_file.write(event_sent_dict[eiid_ev1] + "\n")
                    if not is_ev2_file_exists:
                        dep_ev2_file.write(event_sent_dict[eiid_ev2] + "\n")
                if not is_ev1_file_exists:
                    dep_ev1_file.close()
                if not is_ev2_file_exists:
                    dep_ev2_file.close()


            else:
                print "Files exist"





    def get_max_number_words_in_path(self):
        file_list = self._get_list_of_ev1_ev2_files(self.raw_text_path)
        len_of_dep_context = []
        for f in file_list:
            fname = f+self.parse_sen_file_extn
            parsed_dep_sent_path = os.path.join(self.raw_text_path, fname)
            print "\n File : ",fname
            if not os.path.exists(parsed_dep_sent_path):
                print "File doesn't exist"
                print 20 * "===="
                continue
            with open(parsed_dep_sent_path, "r") as f:
                raw_data = f.readlines()
            for data in raw_data:
                sent = data.strip().split()
                len_of_dep_context.append(len(sent))

            print 20*"===="

        fdist = nltk.FreqDist(len_of_dep_context)

        # Output top 50 words

        for word, frequency in fdist.most_common(50):
            print(u'{};{}'.format(word, frequency))


    def check_and_correct(self):
        file_list = self._get_list_of_ev1_ev2_files(self.raw_text_path)
        missing_files = []
        for f in file_list:
            fname = f+self.parse_sen_file_extn
            parsed_dep_sent_path = os.path.join(self.raw_text_path, fname)

            if not os.path.exists(parsed_dep_sent_path):
                missing_files.append(fname)
                print 20 * "===="
                print "\n File : ", fname
                print "File doesn't exist"
                print 20 * "===="
                continue

            with open(os.path.join(self.raw_text_path,f), "r") as f1:
                raw_data = f1.readlines()
            with open(parsed_dep_sent_path, "r") as f1:
                dep_sent_raw_data = f1.readlines()

            if len(raw_data) != len(dep_sent_raw_data):
                print 20 * "===="
                print "\n File : ", fname
                print "++++ERROR++++"
                print "Re-generating file"
                # self.generate_dep_sentences_file(f)
                os.remove(parsed_dep_sent_path)
                self.generate_for_all_files_from_events([f[:-4]+self.event_file_extn])


            # print 20 * "===="

        fdist = nltk.FreqDist(self.len_of_dep_context)

        # Output top 50 words
        print missing_files

        for word, frequency in fdist.most_common(50):
            print(u'{};{}'.format(word, frequency))





if __name__ == '__main__':
    raw_text_path = "/home/magnet/onkarp/Data/temporal_relations/processed_data/raw_text"
    cps = CreateParsedSentence(raw_text_path)
    # cps.generate_for_all_files_directly()
    # cps.generate_for_all_files_from_events()
    # cps.get_max_number_words_in_path()
    cps.check_and_correct()


