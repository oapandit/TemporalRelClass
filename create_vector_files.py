import numpy as np
import gensim
from shutil import copy2
import string
from utils import *
import io

chars = list(string.ascii_lowercase)
chars.append("num")
chars.append("sym")
character_dict = {k: v + 1 for v, k in enumerate(chars)}

class create_vector_files:
    def __init__(self):
        self.raw_text_path = processed_text_data_path
        self.op_path = processed_vec_data_path
        create_dir(self.op_path)
        self.unknown_words_vec_dict = None
        self.unknown_words_vec_dict_file = "unk_word2vec_dict.pkl"
        self.common_files_path = processed_data_path

        self._num_relations = 6
        self.num_point_relations = 3
        self.word_vec_size = 300

        self.word_context_length = 9
        self.max_chars_in_word = 10

        self.event_head_vec_file_extn = event_head_vec_file_extn

        self.tense_position_in_raw_file = 0
        self.pos_position_in_raw_file = 1
        self.event_num_position_in_raw_file = 6
        self.line_num_position_in_raw_file = 7
        self.is_dep_sent_file= False


        self._relations_dict = {
            'AFTER': 1,
            'BEFORE': 2,
            'IS_INCLUDED': 3,
            'DURING': 3,
            'INCLUDES': 4,
            'DURING_INV': 4,
            'IDENTITY': 5,
            'SIMULTANEOUS': 5,
            'VAGUE':6
        }

        self._reverse_relations_dict = {
            'AFTER': 'BEFORE',
            'BEFORE':'AFTER' ,
            'IS_INCLUDED': 'DURING_INV',
            'DURING': 'DURING_INV',
            'INCLUDES': 'DURING',
            'DURING_INV': 'DURING',
            'IDENTITY': 'IDENTITY',
            'SIMULTANEOUS': 'SIMULTANEOUS',
            'BEGINS': 'BEGUN_BY',
            'ENDS':'ENDED_BY',
            'ENDED_BY': 'ENDS',
            'BEGUN_BY': 'BEGINS',
            'IBEFORE': 'IAFTER',
            'IAFTER': 'IBEFORE',
            'OVERLAP': 'OVERLAP',
            'VAGUE':'VAGUE'
        }

        self._end_point_relation_dict = {
            'AFTER': [0,0,0,0],
            'BEFORE': [1,1,1,1],
            'DURING': [0,1,1,0],
            'IS_INCLUDED': [0,1,1,0],
            'INCLUDES': [1,0,1,0],
            'DURING_INV': [1,0,1,0],
            'IDENTITY': [2,2,1,0],
            'SIMULTANEOUS': [2,2,1,0],
            # 'BEGINS': [1,2,3,1],
            # 'ENDS':[1,2,3,1],
            # 'ENDED_BY': [1,2,3,1],
            # 'BEGUN_BY': [1,2,3,1],
            # 'IBEFORE': [1,2,3,1],
            # 'IAFTER': [1,2,3,1],
            # 'OVERLAP': [1,2,3,1],
            'VAGUE':[1,2,1,2]
        }

    def handle_unknown_words(self, word):
        fname = self.unknown_words_vec_dict_file
        if word == PAD_WORD:
            return np.zeros(self.word_vec_size)
        if self.unknown_words_vec_dict is None:
            logger.debug( "Dict is none")
            if os.path.isfile(os.path.join(self.common_files_path, fname)):
                logger.debug( "Dict file exist")
                with open(os.path.join(self.common_files_path, fname), 'rb') as f:
                    self.unknown_words_vec_dict = pickle.load(f)
            else:
                logger.debug( "Dict file does not exist")
                self.unknown_words_vec_dict = {}
        if self.unknown_words_vec_dict.get(word, None) is not None:
            # logger.debug( "word present in dictionary : ", word
            vec = self.unknown_words_vec_dict.get(word, None)
        else:
            # logger.debug( "word is not present in dictionary : ", word
            vec = np.random.rand(1, self.word_vec_size)
            self.unknown_words_vec_dict[word] = vec
        return vec

    def _char_vec(self,word):

        word = word.lower()

        character_dict = create_vector_files.character_dict

        word_length = 0
        # char_array = np.array([])
        char_num_array = np.array([])
        vec_dim = len(character_dict.keys())
        for ch in word:
            word_length = word_length + 1
            # char_vec = np.zeros(vec_dim)
            if (word_length > self.max_chars_in_word):
                break
            if ch.isdigit():
                ch = 'num'
            if not ch.isalpha():
                ch = 'sym'
            # vec = int(character_dict[ch]) - 1
            # char_vec[vec] = 1
            char_num_array = np.append(char_num_array, int(character_dict[ch]))
            # char_array = np.append(char_array, char_vec)
        if (word_length < self.max_chars_in_word):
            # char_vec = np.zeros(vec_dim)
            while (word_length != self.max_chars_in_word):
                char_num_array = np.append(char_num_array, 0)
                # char_array = np.append(char_array, char_vec)
                word_length = word_length + 1
        # return (char_num_array, char_array)
        return char_num_array

    def get_vec_for_word(self, model, word):
        try:
            if word == PAD_WORD:
                return np.zeros(self.word_vec_size)
            if model is not None:
                vec = model[word]
                return vec
            else:
                # logger.debug(("returning char vector")
                return self._char_vec(word)
        except:
            logger.debug( "Vector not in model for word {}".format(word))
            vec = self.handle_unknown_words(word)
            return vec

    def write_vecs_to_file(self, raw_data_content, vec_file, is_reverse_relation=False, model=None, is_word=True,is_event_head_vec=False):

        def __split_and_get_event_words(sentence):
            sent = sentence.strip()
            if not self.is_dep_sent_file:
                sent = sent.split(feat_separator)[5]
            return sent

        all_vec_array = np.array([])
        if not is_word:
            logger.debug( "Writing rel2vec file")
            for sent in raw_data_content:
                word = sent.strip().upper()
                # if self._relations_dict.get(word,None) is None:
                #     self._relations_dict[word] = self._num_relations+1
                #     self._num_relations +=1
                vec = [0] * self._num_relations
                if is_reverse_relation:
                    vec[self._relations_dict[self._reverse_relations_dict[word]]-1]=1
                else:
                    vec[self._relations_dict[word]-1] =1
                all_vec_array = np.append(all_vec_array, vec)
        else:
            logger.debug( "Writing word2vec file")
            # logger.debug(("number of events : {0}".format(len(raw_data_content)))
            for sent in raw_data_content:

                sent = __split_and_get_event_words(sent)
                # words = word_tokenize(sent)
                words = sent.split()
                if len(words) ==1 and words[0] == PAD_WORD:
                    self.error_counter+=1
                if len(words) != self.word_context_length:
                    # logger.debug( "####ERROR###"
                    # logger.debug( "WORDS of length is not eqaul to context length",len(words)
                    # logger.debug((words)
                    # logger.debug( "############"
                    if len(words) > self.word_context_length:
                        words = words[0:self.word_context_length]
                    else:
                        while len(words)!=self.word_context_length:
                            words.append(PAD_WORD)

                #while creating character vectors; only consider event head word
                # logger.debug(("--"*5)
                # logger.debug((words[4])
                if is_event_head_vec:
                    vec = self.get_vec_for_word(model, words[4])
                    all_vec_array = np.append(all_vec_array, vec)
                else:
                    if model is None:
                        vec = self.get_vec_for_word(model, words[4])
                        all_vec_array = np.append(all_vec_array, vec)
                        # logger.debug((vec)
                    else:
                        for word in words:
                            word = word.strip().lower()
                            vec = self.get_vec_for_word(model, word)
                            all_vec_array = np.append(all_vec_array, vec)
        pickle.dump(all_vec_array, vec_file)
        vec_file.close()


    def write_dep_parsed_vec_file(self,model, raw_data_content, vec_file,num_words_in_sent):
        for sent in raw_data_content:
            words = sent.strip().split()
            while len(words)!=num_words_in_sent:
                words.append(PAD_WORD)
            for word in words:
                word = word.strip().lower()
                vec = self.get_vec_for_word(model, word)
                all_vec_array = np.append(all_vec_array, vec)
        pickle.dump(all_vec_array, vec_file)
        vec_file.close()





    def write_feat_file(self, raw_data_content, vec_file,feat_position,feat_number_dict,unique_feats):

        def __split_and_get_event_specific_feat(sentence):
            sent = sentence.strip()
            sent = sent.split(feat_separator)[feat_position].upper()
            return sent

        all_vec_array = np.array([])
        # logger.debug( "Writing specific feature {} file".format(feat_position)
        for sent in raw_data_content:
            feature = __split_and_get_event_specific_feat(sent)
            if feat_number_dict is not  None:
                if feat_number_dict.get(feature,None) is None:
                    # logger.debug(("{} this feature  is new to me ".format(feature))
                    feat_number_dict[feature] = unique_feats
                    vec = unique_feats
                    unique_feats+=1
                    # logger.debug((unique_feats)

                else:
                    vec = feat_number_dict[feature]
            else:
                vec = int(feature) # for event number and line number adding those ints as it is
            all_vec_array = np.append(all_vec_array, vec)
        pickle.dump(all_vec_array, vec_file)
        vec_file.close()

        return feat_number_dict,unique_feats

    def end_point_file_creator(self,raw_data_content, file,is_reverse_generate):

        if is_reverse_generate:
            if os.path.exists(os.path.join(self.op_path, file[:-3] + "rev_end_point1")):
                return
            end_point1_file = open(os.path.join(self.op_path, file[:-3] + "rev_end_point1"), "wb")
            end_point2_file = open(os.path.join(self.op_path, file[:-3] + "rev_end_point2"), "wb")
            end_point3_file = open(os.path.join(self.op_path, file[:-3] + "rev_end_point3"), "wb")
            end_point4_file = open(os.path.join(self.op_path, file[:-3] + "rev_end_point4"), "wb")

        else:
            if os.path.exists(os.path.join(self.op_path, file[:-3] + "end_point1")):
                return
            end_point1_file = open(os.path.join(self.op_path, file[:-3] + "end_point1"), "wb")
            end_point2_file = open(os.path.join(self.op_path, file[:-3] + "end_point2"), "wb")
            end_point3_file = open(os.path.join(self.op_path, file[:-3] + "end_point3"), "wb")
            end_point4_file = open(os.path.join(self.op_path, file[:-3] + "end_point4"), "wb")

        end_point_files = [end_point1_file,end_point2_file,end_point3_file,end_point4_file]

        all_vec_array = [0]*4

        for sent in raw_data_content:
            word = sent.strip().upper()
            if is_reverse_generate:
                point_list = self._end_point_relation_dict[self._reverse_relations_dict[word]]
            else:
                point_list = self._end_point_relation_dict[word]
            for index,point in enumerate(point_list):
                vec = [0] * self.num_point_relations
                vec[point] = 1
                all_vec_array[index] = np.append(all_vec_array[index], vec)

        for index,file in enumerate(end_point_files):
            all_vec_array[index] = np.delete(all_vec_array[index], [0], None)
            pickle.dump(all_vec_array[index], file)
            file.close()



    def generate_vector_for_all(self):

        self.pos_number_dict = {}
        self.tense_number_dict = {}
        # self.event_num_number_dict = {}
        # self.line_num_number_dict = {}

        self.unique_tense = 1
        self.unique_pos = 1
        # self.unique_eve_num = 1
        # self.unique_line_num = 1

        logger.debug("Loading word2vec model.....")

        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
        logger.debug("Loading complete.")
        files = get_list_of_files_in_dir(self.raw_text_path)
        for file in files:
            if not (file.endswith('.events') and file.endswith('.ev_pairs')):
                logger.debug("Generating vectors for file :{} ".format(file))
                logger.debug(10*"----")
                with open(os.path.join(self.raw_text_path, file), "r") as f:
                    raw_data_content = f.readlines()

                if len(raw_data_content)>0:
                    if file.endswith(".rel"):
                        vec_file_name = os.path.join(self.op_path, file + "_vec")
                        if not os.path.exists(vec_file_name):
                            vec_file = open(vec_file_name, "wb")
                            self.write_vecs_to_file(raw_data_content, vec_file,is_reverse_relation=False,is_word=False)

                        #Adding reverse relation vector file
                        rev_vec_file_name = os.path.join(self.op_path, file + "_rev_vec")
                        if not os.path.exists(rev_vec_file_name):
                            rev_vec_file = open(rev_vec_file_name, "wb")
                            self.write_vecs_to_file(raw_data_content, rev_vec_file, is_reverse_relation=True,is_word=False)

                        #create endpoint relations
                        self.end_point_file_creator(raw_data_content, file,is_reverse_generate=False)
                        #create reverse endpoint relations
                        self.end_point_file_creator(raw_data_content, file, is_reverse_generate=True)
                    if file.endswith(".ev1") or file.endswith(".ev2") or file.endswith(".rand_ev1") or file.endswith(".rand_ev2") :
                        vec_file_name = os.path.join(self.op_path, file + "_vec")
                        if not os.path.exists(vec_file_name):
                            vec_file = open(vec_file_name, "wb")
                            self.write_vecs_to_file(raw_data_content, vec_file, model=model)

                        char_vec_file_name = os.path.join(self.op_path, file + "_char_vec")
                        if not os.path.exists(char_vec_file_name):
                            char_vec_file = open(char_vec_file_name, "wb")
                            self.write_vecs_to_file(raw_data_content, char_vec_file)

                        event_head_vec_file_name = os.path.join(self.op_path, file +"_vec" + self.event_head_vec_file_extn)
                        if not os.path.exists(event_head_vec_file_name):
                            event_head_vec_file = open(event_head_vec_file_name, "wb")
                            self.write_vecs_to_file(raw_data_content, event_head_vec_file,model=model,is_event_head_vec=True)

                        tense_vec_file_name = os.path.join(self.op_path, file + tense_file_extn)
                        if not os.path.exists(tense_vec_file_name):
                            tense_vec_file = open(tense_vec_file_name, "wb")
                            self.tense_number_dict, self.unique_tense = self.write_feat_file(raw_data_content, tense_vec_file,self.tense_position_in_raw_file,self.tense_number_dict,self.unique_tense)

                        pos_vec_file_name = os.path.join(self.op_path, file + pos_file_extn)
                        if not os.path.exists(pos_vec_file_name):
                            pos_vec_file = open(pos_vec_file_name, "wb")
                            self.pos_number_dict, self.unique_pos = self.write_feat_file(raw_data_content, pos_vec_file,self.pos_position_in_raw_file,self.pos_number_dict,self.unique_pos)

                        event_num_vec_file_name = os.path.join(self.op_path, file + event_num_file_extn)
                        if not os.path.exists(event_num_vec_file_name):
                            event_num_vec_file = open(event_num_vec_file_name, "wb")
                            self.write_feat_file(raw_data_content, event_num_vec_file,
                                                 self.event_num_position_in_raw_file, None,None)

                        line_num_vec_file_name = os.path.join(self.op_path, file + line_num_file_extn)
                        if not os.path.exists(line_num_vec_file_name):
                            line_num_vec_file = open(line_num_vec_file_name, "wb")
                            self.write_feat_file(raw_data_content, line_num_vec_file,
                                                 self.line_num_position_in_raw_file, None,None)




                if file.endswith(".ev_pairs") or file.endswith(".rand_ev_pairs"):
                    copy2(os.path.join(self.raw_text_path, file), os.path.join(self.op_path, file))

        if self.unknown_words_vec_dict is not None:
            logger.debug("saving final unknown word2vec dictionary to file")
            f = open(os.path.join(self.common_files_path, self.unknown_words_vec_dict_file), "wb")
            pickle.dump(self.unknown_words_vec_dict, f)
            f.close()


    def load_fasttext_char_vectors(self):
        logger.debug("Loading fasttext character vector file...")
        fin = io.open(fasttext_model_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
        logger.debug("fasttext vectors loaded. Returning ...")
        return data

    def fasttext_char_emb_files(self):
        files = get_list_of_files_in_dir(self.raw_text_path)
        model = self.load_fasttext_char_vectors()
        for file in files:
            if file.endswith(".ev1") or file.endswith(".ev2"):
                logger.debug("File : {}".format(file))
                with open(os.path.join(self.raw_text_path, file), "r") as f:
                    raw_data_content = f.readlines()
                fasttext_char_vec_file_name = os.path.join(self.op_path, file + "_fasttext_char_all_cont_vec")
                if not os.path.exists(fasttext_char_vec_file_name):
                    event_head_vec_file = open(fasttext_char_vec_file_name, "wb")
                    self.write_vecs_to_file(raw_data_content, event_head_vec_file, model=model,
                                            is_event_head_vec=False)

                logger.debug(20 * "***")



    def generate_point_rel_vectors(self):
        files = get_list_of_files_in_dir(self.raw_text_path)
        for file in files:
            if not file.endswith('.events'):
                logger.debug("File : {}".format(file))
                if file.endswith(".rel"):
                    with open(os.path.join(self.raw_text_path, file), "r") as f:
                        raw_data_content = f.readlines()
                    self.end_point_file_creator(raw_data_content, file,is_reverse_generate=False)
                    self.end_point_file_creator(raw_data_content, file, is_reverse_generate=True)
                logger.debug(20 * "***")




    def generate_vectors_only_for_rand_pairs(self):
        logger.debug("Loading word2vec model.....")
        # model = Word2Vec.load_word2vec_format(os.path.join(word2vec_model_path, word2vec_model_name), binary=True)
        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
        logger.debug("Loading complete.")
        files = get_list_of_files_in_dir(self.raw_text_path)
        for file in files:
            if file.endswith(".rand_ev1") or file.endswith(".rand_ev2") :
                logger.debug("File : {}".format(file))
                with open(os.path.join(self.raw_text_path, file), "r") as f:
                    raw_data_content = f.readlines()
                vec_file_name = os.path.join(self.op_path, file + "_vec")
                if not os.path.exists(vec_file_name):
                    vec_file = open(vec_file_name, "wb")
                    self.write_vecs_to_file(raw_data_content, vec_file, model=model)

                char_vec_file_name = os.path.join(self.op_path, file + "_char_vec")
                if not os.path.exists(char_vec_file_name):
                    char_vec_file = open(char_vec_file_name, "wb")
                    self.write_vecs_to_file(raw_data_content, char_vec_file)
            logger.debug(20 * "***")


        # logger.debug("saving final unknown word2vec dictionary to file"
        # f = open(os.path.join(self.common_files_path, self.unknown_words_vec_dict_file), "wb")
        # pickle.dump(self.unknown_words_vec_dict, f)
        # f.close()



    def generate_vectors_for_dep_sent(self):
        self.word_context_length = 7
        self.is_dep_sent_file =True
        self.error_counter =0
        logger.debug("Loading word2vec model.....")
        # model = Word2Vec.load_word2vec_format(os.path.join(word2vec_model_path, word2vec_model_name), binary=True)
        model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
        logger.debug("Loading complete.")
        files = get_list_of_files_in_dir(self.raw_text_path)
        for file in files:
            if file.endswith(".ev1_dep_parse_words") or file.endswith(".ev2_dep_parse_words") :
                logger.debug("File : {}".format(file))
                with open(os.path.join(self.raw_text_path, file), "r") as f:
                    raw_data_content = f.readlines()
                vec_file_name = os.path.join(self.op_path, file + "_vec")

                path = search_single_file_in_dir(self.op_path,file + "_vec")

                if path is None:
                    vec_file = open(vec_file_name, "wb")
                    self.write_vecs_to_file(raw_data_content, vec_file, model=model)
                else:
                    if os.path.getmtime(path)<os.path.getmtime(os.path.join(self.raw_text_path, file)):
                        logger.debug("File exists; but words file modified after vector creation.")
                        vec_file = open(vec_file_name, "wb")
                        self.write_vecs_to_file(raw_data_content, vec_file, model=model)

            logger.debug(20 * "***")


        self.word_context_length = 9
        self.is_dep_sent_file =False


if __name__ == "__main__":

    # Generate vectors for reduced relations
    gen_vec_files = create_vector_files()
    gen_vec_files.generate_vector_for_all()
    gen_vec_files.fasttext_char_emb_files()


    # data_separator(VEC_FILES_PATH).segrate_docs()
