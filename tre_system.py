import numpy as np
from read_vec_data import read_vec_data
from models import models
import itertools
# from result import EvaluationResult
from utils import *
from sklearn.utils import class_weight
from sklearn.linear_model import LogisticRegression

# fix random seed for reproducibility
seed = 1321
np.random.seed(seed)

os.environ["TMPDIR"] = "/home/tmp"  # For optimization; require enough space

NB_EPOCH = 5
BATCH_SIZE = 50

EXPT_NAME = "expt1_w2vec_fasttext_concate_expts"
# EXPT_NAME = None

class tre_system:
    def __init__(self):
        self.word_vector_size = WORD_VECTOR_SIZE
        self.char_vector_size = CHAR_VECTOR_SIZE
        self.word_context_length = WORD_CONTEXT_LENGTH
        self.num_interval_relations = NUM_INTERVAL_RELATIONS
        self.num_point_relations = NUM_POINT_RELATIONS
        self.vec_files_path = VEC_FILES_PATH
        self.train_data_path = os.path.join(self.vec_files_path, "train_data")
        self.dev_data_path = os.path.join(self.vec_files_path, "dev_data")
        self.test_data_path = os.path.join(self.vec_files_path, "test_data")
        self.models_path = os.path.join(SAVED_MODEL_PATH)
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        self.mcnemar_stat_list = []
        # self.maitain_consistency = maintain_consistency()
        # self.result = EvaluationResult(EXPT_NAME)
        self.model = models(self.word_context_length, self.word_vector_size,self.char_vector_size,self.num_interval_relations,self.num_point_relations)
        self.data = None # to avoid reading data multiple times
        self.test_file_list = None

        self.expt_name = "EXPT1_"
        self.train_data_extn,self.dev_data_extn,self.test_data_extn = EXPT_1
        self.combination_scheme = None
        self.is_dep_sent_expt = False
        self.ev_pairs_file_extn = ".ev_pairs"
        self.is_fasttext = False
        self.w2vec_char_concate = False

    def _get_class_weights(self,y):
        y = np.argmax(y,axis=1)
        unique, counts = np.unique(y, return_counts=True)
        # class_occ = dict(zip(unique, counts))
        for i in range(len(unique)):
            print("{0} \t {1}".format(unique[i],counts[i]))
        class_weights = class_weight.compute_class_weight('balanced',unique,y)
        print(class_weights)
        return class_weights


    def _generate_train_model_name_from_params(self, hyper_params, is_point_relation=True,is_return_file_name=False):
        '''
        returns filename for trained model based on hyperparams used and flag is_point_relation
        '''
        hyper_params = list(hyper_params)
        hyper_params.append(NB_EPOCH)
        hyper_params.append(BATCH_SIZE)
        if is_point_relation:
            model_fname = self.expt_name+"point_rel_{0}_interaction_{1}_char_{2}_input_drop_out_{3}_num_rnn_neurons_{4}_cnn_filters_{5}_opt_{6}_epoch_{7}_batch_{8}.h5".format(
                *hyper_params)
        else:
            model_fname = self.expt_name+"interval_rel_{0}_interaction_{1}_char_{2}_input_drop_out_{3}_num_rnn_neurons_{4}_cnn_filters_{5}_opt_{6}_epoch_{7}_batch_{8}_tmax.h5".format(
                *hyper_params)

        if is_return_file_name:
            return model_fname
        else:
            print("model name : ", model_fname)
            return os.path.join(self.models_path, model_fname)

    def _get_combinations_params(self,is_list=False,is_diff_interactions=False,is_char=False,is_pos=False):
        '''
        This subroutine returns all combinations of hyperparameters used for training.
        If for training, just want fixed valued params, is_list should be false.
        :param is_list:
        :return: list of tuples
        '''
        if is_diff_interactions:
            interaction_list = ["ADDITION", "SUBTRACTION", "MULTIPLICATION", "CONCATE", "MLP", "CNN"]
        else:
            interaction_list = ["DEEP_CNN"]
        if is_list:
            nns = ["RNN", "GRU", "LSTM", "BRNN", "BGRU", "BLSTM"]
            optimizers = ['adam', 'sgd', 'rmsprop']
            num_rnn_neurons_list = [128]
            input_drop_out_list = [0.4]
            cnn_filters_list = [64]
            num_rnn_neurons_list = [16, 32, 64, 128, 256]
            input_drop_out_list = [0.2, 0.3, 0.4, 0.5]
            cnn_filters_list = [32, 64, 128, 256]

            # num_rnn_neurons_list = [16, 64, 256]
            # input_drop_out_list = [0.1, 0.3, 0.5]
            # cnn_filters_list = [32, 64, 128]
        else:
            nns = ["BGRU"]
            optimizers = ['adam']
            num_rnn_neurons_list = [128]
            input_drop_out_list = [0.4]
            cnn_filters_list = [64]

        a = []
        a.append(interaction_list)
        a.append([is_char])
        a.append([is_pos])
        a.append(nns)
        a.append(input_drop_out_list)
        a.append(num_rnn_neurons_list)
        a.append(cnn_filters_list)
        a.append(optimizers)
        list_of_all_params_combinations = list(itertools.product(*a))
        return list_of_all_params_combinations


    def _combine_event_head_data(self,data):
        self.combination_scheme= "CONCATE" if self.combination_scheme is None else self.combination_scheme

        if self.combination_scheme == "SUM":
            data = np.add((data[0],data[1]))

        elif self.combination_scheme == "SUBTRACT":
            data = np.subtract((data[0], data[1]))

        else:
            data = np.concatenate((data[0], data[1]), axis=1)

        return data



    def _read_vec_data(self,is_test_only=False,is_point_data=False,is_char= False,is_pos=False,is_test_on_rand_pairs=False,is_train_without_vague=False,is_load_only_event_head_data=False):
        '''
        read vector data for training, dev and test.
        if only test data to be read : is_test_only = True. In this case train and dev data would be none.
                                        else read all three
        if point data to be read : is_point_data should be true else interval relation data would be read.
        :param is_test_only:
        :param is_point_data:
        :return:
        '''
        # if is_load_only_event_head_data:
        #     self.word_context_length = 1
        print(self.is_dep_sent_expt)
        print("Loading data ....")
        data = []
        if is_test_only:
            event1_train, event2_train, relation_train = [None] * 3
            event1_dev, event2_dev, relation_dev = [None] * 3
            test_data = read_vec_data(self.test_data_path, self.num_interval_relations, self.num_point_relations,
                                      self.word_context_length, self.word_vector_size,
                                      self.char_vector_size,is_test_on_rand_pairs,data_set_to_load=self.test_data_extn,is_load_only_event_head_data=is_load_only_event_head_data)  # loading test data
            if self.is_fasttext:
                test_data.is_fasttext = True

            if self.w2vec_char_concate:
                test_data.w2vec_char_concate = True
            if self.is_dep_sent_expt:
                test_data.load_dep_sent_vecs = ""
            test_file_list = test_data.get_file_name_list()
            # event1_test, event2_test, relation_test = test_data.load_data(load_reverse_data=False,load_point_relations=is_point_data,load_char_data=is_char)
            event_test, relation_test = test_data.load_data(load_reverse_data=False,
                                                            load_point_relations=is_point_data,
                                                            load_char_data=is_char,is_train_without_vague=is_train_without_vague,load_pos_data=is_pos)
            train_data =None
            dev_data = None

        else:
            train_data = read_vec_data(self.train_data_path, self.num_interval_relations, self.num_point_relations,
                                       self.word_context_length, self.word_vector_size,self.char_vector_size,data_set_to_load=self.train_data_extn,is_load_only_event_head_data=is_load_only_event_head_data)

            if self.is_fasttext:
                train_data.is_fasttext = True

            if self.w2vec_char_concate:
                train_data.w2vec_char_concate = True
            if self.is_dep_sent_expt:
                # print("setting dep sent extns")
                train_data.load_dep_sent_vecs= ""

            event_train, relation_train = train_data.load_data(load_reverse_data=True,load_point_relations=is_point_data,load_char_data=is_char,is_train_without_vague=is_train_without_vague,load_pos_data=is_pos)
            # print("\n".join(train_data.get_file_name_list()))

            dev_data = read_vec_data(self.dev_data_path, self.num_interval_relations, self.num_point_relations,
                                     self.word_context_length, self.word_vector_size,self.char_vector_size,data_set_to_load=self.dev_data_extn,is_load_only_event_head_data=is_load_only_event_head_data)

            if self.is_fasttext:
                dev_data.is_fasttext = True

            if self.w2vec_char_concate:
                dev_data.w2vec_char_concate = True
            if self.is_dep_sent_expt:
                # print("setting dep sent extns")
                dev_data.load_dep_sent_vecs= ""
            event_dev,relation_dev = dev_data.load_data(load_reverse_data=False,
                                                                  load_point_relations=is_point_data,
                                                                  load_char_data=is_char,is_train_without_vague=is_train_without_vague,load_pos_data=is_pos)
            # print("\n".join(dev_data.get_file_name_list()))
            test_file_list = dev_data.get_file_name_list()
            event_test = event_dev
            relation_test = relation_dev

            train_data = [event_train, relation_train]
            dev_data = [event_dev, relation_dev]
        test_data = [event_test, relation_test]
        data = [train_data,dev_data,test_data]
        # print(test_file_list)
        # self.word_context_length = WORD_CONTEXT_LENGTH # restoring value if changed because of loading only event head data
        return data,test_file_list

    def _test_model(self,is_point_relation,hyper_params,model,is_char,is_test_on_rand_pairs,is_train_without_vague):

        model_file_path = self._generate_train_model_name_from_params(hyper_params,is_point_relation)

        # get neural model based on flag : is_point_relation
        print("getting model .....")
        if self.is_dep_sent_expt:
            model.word_context_length = 7
        if is_train_without_vague:
            model.num_interval_relations = 5
        test_model = model.get_neural_model(is_point_relation,hyper_params)

        if not os.path.isfile(model_file_path):
            print("Model file {0} does exist. " .format(model_file_path.split('/')[-1]))

        else:
            # read data only once
            if self.data is None:
                self.data, self.test_file_list = self._read_vec_data(is_test_only=True, is_point_data=is_point_relation,is_char=is_char,is_test_on_rand_pairs=is_test_on_rand_pairs,is_train_without_vague=is_train_without_vague)
            print("Model file {0} exists".format(model_file_path.split('/')[-1]))
            test_model.load_weights(model_file_path)

            relation_gold = self.data[2][1]
            relation_pred = test_model.predict(self.data[2][0])

        return relation_gold,relation_pred,self.test_file_list

    def _train_model(self, force_train, is_point_relation, hyper_params, model, is_char,is_pos, is_test_on_rand_pairs,is_train_without_vague,is_logistic_model=False):

        if hyper_params is not None:
            model_file_path = self._generate_train_model_name_from_params(hyper_params,is_point_relation)
        else:
            model_file_path = os.path.join(self.models_path, self.model_file_name)

        # get neural model based on flag : is_point_relation
        print("getting model .....")
        if self.is_dep_sent_expt:
            model.word_context_length = 7
        if not is_logistic_model:
            if is_train_without_vague:
                model.num_interval_relations = 5
            if self.is_fasttext:
                model.is_fasttext=True
            if self.w2vec_char_concate:
                model.word_vector_size = self.word_vector_size + 300
            train_model = model.get_neural_model(is_point_relation,hyper_params)
        else:
            train_model = model

        relation_gold, relation_pred = None,None

        if not os.path.isfile(model_file_path) or force_train:

            # read data only once
            if self.data is None:
                self.data, self.test_file_list = self._read_vec_data(is_test_only=False, is_point_data=is_point_relation,is_char=is_char,is_pos=is_pos,is_test_on_rand_pairs=is_test_on_rand_pairs,is_train_without_vague=is_train_without_vague,is_load_only_event_head_data=is_logistic_model)

            data = self.data
            print("learning model params .....")
            if is_logistic_model:
                if len(data[0][0])==2:
                    data[0][0] = self._combine_event_head_data(data[0][0])
                    data[1][0] = self._combine_event_head_data(data[1][0])

                    data[0][1] = np.argmax(data[0][1], axis=1)
                    data[1][1] = np.argmax(data[1][1], axis=1)
                train_model.fit(data[0][0],data[0][1])
            else:
                train_model.fit(data[0][0],data[0][1], epochs=NB_EPOCH, batch_size=BATCH_SIZE,validation_data=(data[1][0],data[1][1]),class_weight=self._get_class_weights(data[0][1]))
            if not is_logistic_model:
                train_model.save_weights(model_file_path)
                print("Saved model to file : " + model_file_path.split('/')[-1])
            else:
                save_as_pickle_file(train_model,model_file_path)
            relation_gold = data[1][1]
            relation_pred = train_model.predict(data[1][0])

        else:
            print("Model file {0} exists. Not training....".format(model_file_path.split('/')[-1]))
            # train_model.load_weights(model_file_path)



        return relation_gold,relation_pred,self.test_file_list



    def train_neural_model(self):
        is_point_relation = False
        force_train = True
        is_char = False
        is_test_on_rand_pairs = False
        train_without_vauge= False
        is_pos = False
        self.result.train_without_vauge = train_without_vauge
        self.is_fasttext=False
        self.w2vec_char_concate = False
        # as we are passing False for both the flags, we will get list with only one tuple
        # for ii in range(10):
        hyper_params = self._get_combinations_params(is_list=False, is_diff_interactions=False,is_char=is_char,is_pos=is_pos)[0]
        model_file_name = self._generate_train_model_name_from_params(hyper_params, is_point_relation,
                                                                      is_return_file_name=True)
        for _ in range(1):
            relation_gold,relation_pred, test_file_list = self._train_model(force_train=force_train, is_point_relation=is_point_relation,
                                                                            hyper_params=hyper_params, model=self.model, is_char=is_char,is_pos=is_pos, is_test_on_rand_pairs=is_test_on_rand_pairs,is_train_without_vague=train_without_vauge)

            # relation_gold, relation_pred, test_file_list = self._test_model( is_point_relation=is_point_relation,
            #                                                                  hyper_params=hyper_params, model=self.model,
            #                                                                  is_char=is_char,is_pos=is_pos,
            #                                                                  is_test_on_rand_pairs=is_test_on_rand_pairs,
            #                                                                  is_train_without_vague=train_without_vauge)

            if relation_gold is not None:
                self.result.create_proba_score_file(relation_pred)
                # f_list = ['APW19980227.0487_TD.tml', 'CNN19980223.1130.0960_TD.tml', 'NYT19980212.0019_TD.tml', 'PRI19980216.2000.0170_TD.tml', 'ed980111.1130.0089_TD.tml']
                # self.result.evaluate_with_minimal_graph(relation_pred,file_list=test_file_list)
                # self.result.evaluate_direct(relation_gold, relation_pred,is_print_report=True)
                # self.result.evaluate_with_temporal_awareness(relation_pred, test_file_list)
                # self.result.log_scores_in_file(model_file_name)
                # self.maitain_consistency.read_interval_relations_events_and_probability_score(test_file_list,
                #                                                                               relation_pred)
                self.maitain_consistency.read_end_point_relations_events_and_probability_score(test_file_list,relation_pred)


    def train_different_interaction_neural_models(self):
        is_point_relation = False
        force_train = False
        is_char = False
        is_test_on_rand_pairs = False
        train_without_vauge= False
        is_pos = False
        self.result.train_without_vauge = train_without_vauge
        hyper_params_list = self._get_combinations_params(is_list=False,is_diff_interactions=True,is_char=is_char,is_pos=is_pos)
        self.is_fasttext = True
        for hyper_params in hyper_params_list:
            model_file_name = self._generate_train_model_name_from_params(hyper_params, is_point_relation,
                                                                          is_return_file_name=True)
            relation_gold,relation_pred, test_file_list = self._train_model(force_train=force_train, is_point_relation=is_point_relation,
                                                                        hyper_params=hyper_params, model=self.model, is_char=is_char,is_pos=is_pos, is_test_on_rand_pairs=is_test_on_rand_pairs,is_train_without_vague=train_without_vauge)
            # relation_gold, relation_pred, test_file_list = self._test_model( is_point_relation=is_point_relation,
            #                                                                  hyper_params=hyper_params, model=self.model,
            #                                                                  is_char=is_char,is_pos=is_pos,
            #                                                                  is_test_on_rand_pairs=is_test_on_rand_pairs,
            #                                                                  is_train_without_vague=train_without_vauge)
            if relation_gold is not None:
                self.result.evaluate_with_temporal_awareness(relation_pred,test_file_list)
                self.result.evaluate_with_minimal_graph(relation_pred,file_list=test_file_list)
                # self.result.create_proba_score_file(relation_pred)
                self.result.evaluate_direct(relation_gold, relation_pred)
                self.result.log_scores_in_file(model_file_name)

    def get_best_model(self):
        '''
        We will do grid search for all the parameters to get the best performing model.

        :return:
        '''
        force_train = False
        is_char = False
        is_test_on_rand_pairs = False
        train_without_vauge= False
        is_pos = False
        is_point_relation = False
        hyper_params_list = self._get_combinations_params(is_list=True, is_diff_interactions=False,is_char=is_char,is_pos=is_pos)

        self.is_fasttext=False
        self.w2vec_char_concate = True
        # hyper_params_list = hyper_params_list[0:5]
        is_test_on_rand_pairs = False
        counter =1
        self.result.train_without_vauge = train_without_vauge

        for hyper_params in hyper_params_list:
            print("\n"*3)
            print("training {0} model ..... {1} to go.....".format(counter,(len(hyper_params_list)-counter)))
            model_file_name = self._generate_train_model_name_from_params(hyper_params, is_point_relation,is_return_file_name=True)
            relation_gold,relation_pred, test_file_list = self._train_model(force_train=force_train, is_point_relation=is_point_relation,
                                                                        hyper_params=hyper_params, model=self.model, is_char=is_char,is_pos=is_pos, is_test_on_rand_pairs=is_test_on_rand_pairs,is_train_without_vague=train_without_vauge)
            if relation_gold is not None:
                self.result.evaluate_with_temporal_awareness(relation_pred, test_file_list)
                self.result.evaluate_direct(relation_gold,relation_pred)
                self.result.evaluate_with_minimal_graph(relation_pred,file_list=test_file_list)
                self.result.log_scores_in_file(model_file_name)
            counter+=1
        self.result.save_score_dicts()


    def train_rel_logistic_model(self):


        is_point_relation = False
        force_train = False
        is_char = False
        is_test_on_rand_pairs = False
        train_without_vauge= False


        C = [0.001,0.01,0.1,1,10,100]
        fit_ints = [True,False]
        class_wts = [None,"balanced"]
        intercept_scalinngs = [0.001,0.01,0.1,1,10,100]

        a = []
        a.append(C)
        a.append(fit_ints)
        a.append(class_wts)
        a.append(intercept_scalinngs)
        list_of_all_params_combinations = list(itertools.product(*a))

        self.result.train_without_vauge = train_without_vauge
        for params in list_of_all_params_combinations:
            log_model = LogisticRegression(C=params[0],fit_intercept=params[1],class_weight=params[2],intercept_scaling=params[3],multi_class='ovr')

            print(15*"=======")
            # print("TRAINING CONCATENATE LOGISTIC MODEL")
            self.model_file_name = 'concate_logistic_model_C_{0}_fit_{1}_class_{2}_int_{3}.sav'.format(*params)
            relation_gold,relation_pred, test_file_list = self._train_model(force_train=force_train, is_point_relation=is_point_relation,
                                                                            hyper_params=None, model=log_model, is_char=is_char, is_test_on_rand_pairs=is_test_on_rand_pairs,is_train_without_vague=train_without_vauge,is_logistic_model=True)

            # relation_gold, relation_pred, test_file_list = self._test_model( is_point_relation=is_point_relation,
            #                                                                  hyper_params=hyper_params, model=self.model,
            #                                                                  is_char=is_char,
            #                                                                  is_test_on_rand_pairs=is_test_on_rand_pairs,
            #                                                                  is_train_without_vague=train_without_vauge)

            if relation_gold is not None:
                self.result.evaluate_with_minimal_graph(relation_pred,file_list=test_file_list)
                self.result.evaluate_direct(relation_gold, relation_pred,is_print_report=True)
                self.result.evaluate_with_temporal_awareness(relation_pred, test_file_list)
                self.result.log_scores_in_file(self.model_file_name)


    def train_cnn_with_word_emb(self):

        self.model_file_name = os.path.join(self.models_path,"cnn_word_emb.h5")

        is_test_only = True
        is_point_relation = False
        force_train = True
        is_char = False
        is_test_on_rand_pairs = False
        is_train_without_vague= False

        is_logistic_model = True # to get only event head data

        train_model = self.model.get_word_emb_cnn(num_relations=NUM_INTERVAL_RELATIONS)
        self.data, self.test_file_list = self._read_vec_data(is_test_only=is_test_only, is_point_data=is_point_relation,is_char=is_char,is_test_on_rand_pairs=is_test_on_rand_pairs,is_train_without_vague=is_train_without_vague,is_load_only_event_head_data=is_logistic_model)
        # train_model.fit(self.data[0][0],self.data[0][1], epochs=NB_EPOCH, batch_size=BATCH_SIZE,validation_data=(self.data[1][0],self.data[1][1]),class_weight=self._get_class_weights(self.data[0][1]))
        # train_model.save_weights(self.model_file_name)
        train_model.load_weights(self.model_file_name)
        relation_gold = self.data[2][1]
        relation_pred = train_model.predict(self.data[2][0])
        if relation_gold is not None:
            self.result.evaluate_with_minimal_graph(relation_pred, file_list=self.test_file_list)
            self.result.evaluate_direct(relation_gold, relation_pred, is_print_report=True)
            self.result.evaluate_with_temporal_awareness(relation_pred, self.test_file_list)
            self.result.log_scores_in_file(self.model_file_name)


    def train_with_dep_sent(self):

        self.model_file_name = os.path.join(self.models_path,"dep_sent.h5")

        is_test_only = False
        is_point_relation = False
        force_train = True
        is_char = False
        is_test_on_rand_pairs = False
        is_train_without_vague= False
        is_pos = False
        self.is_dep_sent_expt = True
        is_logistic_model = False # to get only event head data
        if self.is_dep_sent_expt:
            self.model.word_context_length = 7

        hyper_params = \
        self._get_combinations_params(is_list=False, is_diff_interactions=False, is_char=is_char, is_pos=is_pos)[0]
        train_model = self.model.get_neural_model(is_point_relation,hyper_params)
        self.data, self.test_file_list = self._read_vec_data(is_test_only=is_test_only, is_point_data=is_point_relation,is_char=is_char,is_test_on_rand_pairs=is_test_on_rand_pairs,is_train_without_vague=is_train_without_vague,is_load_only_event_head_data=is_logistic_model)
        train_model.fit(self.data[0][0],self.data[0][1], epochs=NB_EPOCH, batch_size=BATCH_SIZE,validation_data=(self.data[1][0],self.data[1][1]),class_weight=self._get_class_weights(self.data[0][1]))
        train_model.save_weights(self.model_file_name)
        # train_model.load_weights(self.model_file_name)
        relation_gold = self.data[2][1]
        relation_pred = train_model.predict(self.data[2][0])
        if relation_gold is not None:
            self.result.create_proba_score_file(relation_pred)
            self.result.evaluate_with_minimal_graph(relation_pred, file_list=self.test_file_list)
            self.result.evaluate_direct(relation_gold, relation_pred, is_print_report=True)
            self.result.evaluate_with_temporal_awareness(relation_pred, self.test_file_list)
            self.result.log_scores_in_file(self.model_file_name)



    def train_with_structured_learning(self):

        is_load_only_event_head_data = False

        is_point_data = False
        is_char = False
        is_pos = False
        is_train_without_vague = False
        load_reverse_data = False

        hyper_params = self._get_combinations_params(is_list=False, is_diff_interactions=False, is_char=is_char,is_pos=is_pos)[0]
        # create train data object
        train_data = read_vec_data(self.train_data_path, self.num_interval_relations, self.num_point_relations,
                                   self.word_context_length, self.word_vector_size, self.char_vector_size,
                                   data_set_to_load=self.train_data_extn,
                                   is_load_only_event_head_data=is_load_only_event_head_data)

        dev_data = read_vec_data(self.dev_data_path, self.num_interval_relations, self.num_point_relations,
                                 self.word_context_length, self.word_vector_size, self.char_vector_size,
                                 data_set_to_load=self.dev_data_extn,
                                 is_load_only_event_head_data=is_load_only_event_head_data)
        event_dev, relation_dev = dev_data.load_data(load_reverse_data=False,
                                                     load_point_relations=is_point_data,
                                                     load_char_data=is_char,
                                                     is_train_without_vague=is_train_without_vague,load_pos_data=is_pos)

        train_file_list = train_data.get_file_name_list()
        train_model = self.model.get_neural_model(is_point_data, hyper_params)

        train_data_generator = train_data.data_generator_temp_relations(load_reverse_data=load_reverse_data, load_char_data=is_char)
        # print("\n".join(train_file_list))
        for epoch in range(NB_EPOCH):
            print(epoch)
            for f in train_file_list:
                ev_pairs_file = f[0:f.rfind('.')] + self.ev_pairs_file_extn
                with open(search_single_file_in_dir(RAW_DATA_PATH, ev_pairs_file), "r") as f:
                    ev_pairs = f.readlines()
                self.maitain_consistency.ev_pairs = ev_pairs
                event_train, relation_train = next(train_data_generator)
                loss = train_model.train_on_batch(event_train, relation_train)
                print(loss)

        train_model.save_weights(os.path.join(self.models_path,"train_on_batch.h5"))



if __name__ == "__main__":
    tre_sys = tre_system()
    tre_sys.train_neural_model()
    # tre_sys.get_best_model()
    # tre_sys.train_different_interaction_neural_models()
    # tre_sys.train_rel_logistic_model()
    # tre_sys.train_cnn_with_word_emb()
    # tre_sys.train_with_structured_learning()
    # tre_sys.train_with_dep_sent()