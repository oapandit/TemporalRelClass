import os
import numpy as np
from read_vec_data import read_vec_data
from models import models
from sklearn.metrics import f1_score
import pickle
import operator
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import xml.etree.ElementTree as ET
from models import models
import logging
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

NB_EPOCH = 75
BATCH_SIZE = 150
SAVED_MODEL_PATH = "/home/magnet/onkarp/Results/temporal_relations"
RAW_TEXT_FILES ="/home/magnet/onkarp/Data/temporal_relations/raw_data/te3-platinum"
toolkit_path = "/home/magnet/onkarp/Code/temporal_relations/tempeval3_toolkit"

class tre_logistic_system():
    def __init__(self, vec_files_path, num_relations):
        self.word_vector_size = 300
        self.word_context_length = 1
        self.num_relations = num_relations
        self.vec_files_path = vec_files_path
        self.train_data_path = os.path.join(self.vec_files_path, "train_data")
        self.dev_data_path = os.path.join(self.vec_files_path, "dev_data")
        self.test_data_path = os.path.join(self.vec_files_path, "test_data")

        self.models_path = os.path.join(SAVED_MODEL_PATH, "saved_models")
        self.log_file_name = os.path.join(self.models_path, "report_word_emb_cnn.log")
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)


    def train_rel_logistic_model(self):

        test_only = True
        # def _reshape_event_data(event1,event2):


        def _train_model(X_train,X_dev,model,model_fname,dev_file_list):

            print(model_fname)

            if os.path.exists(os.path.join(self.models_path, model_fname)):
            # if False:
            #     print("Model exists already")
                model = pickle.load(open(os.path.join(self.models_path, model_fname), 'rb'))

            else:
                model = model.fit(X_train, relation_train)
                pickle.dump(model, open(os.path.join(self.models_path, model_fname), 'wb'))


            relation_pred = model.predict(X_dev)

            f_score = f1_score(relation_dev_index, relation_pred, average='weighted')
            print(f_score)
            # print(classification_report(relation_dev_index, relation_pred, digits=5))
            cf_table = confusion_matrix(relation_dev_index, relation_pred)
            # print(cf_table)


            mn_score = mcnemar(cf_table, exact=False, correction=True)
            print(mn_score.statistic)
            print(mn_score.pvalue)
            # self.reuslt_file(dev_file_list,relation_pred)
            # return cf_table

        def _concat_events(event1,event2):
            return np.concatenate((event1,event2), axis=1)


        def _sum_events(event1,event2):
            return np.add(event1,event2)


        def _subtract_events(event1,event2):
            return np.subtract(event2,event1)


        if test_only:
            event1_train, event2_train, relation_train = [None] * 3

        else:

            train_data = read_vec_data(self.train_data_path, self.num_relations, self.word_context_length,
                                   self.word_vector_size)
            event1_train, event2_train, relation_train = train_data.load_data(load_reverse_data=False)
            relation_train = np.argmax(relation_train, axis=1)

        dev_data = read_vec_data(self.dev_data_path, self.num_relations, self.word_context_length,
                                 self.word_vector_size)
        event1_dev, event2_dev, relation_dev = dev_data.load_data(load_reverse_data=False)

        relation_dev_index = np.argmax(relation_dev, axis=1)
        dev_file_list = dev_data.get_file_name_list()


        if test_only:
            X_train_concat = None
            X_train_sum = None
            X_train_subtract = None
        else:
            X_train_concat = _concat_events(event1_train,event2_train)
            X_train_sum = _sum_events(event1_train, event2_train)
            X_train_subtract = _subtract_events(event1_train, event2_train)

        X_dev_concat = _concat_events(event1_dev, event2_dev)
        X_dev_sum = _sum_events(event1_dev, event2_dev)
        X_dev_subtract = _subtract_events(event1_dev, event2_dev)

        if test_only:
            print("TRAINING DATA : NONE")
        else:
            print("EVENT TRAIN _concat " + " : shape : " + str(X_train_concat .shape))
        # print("EVENT DEV _concat " + " : shape : " + str(X_dev_concat .shape))

        C = [0.001,0.01,0.1,1,10,100]
        fit_ints = [True,False]
        class_wts = [None,"balanced"]
        intercept_scalinngs = [0.001,0.01,0.1,1,10,100]

        for c in C:
            for fit in fit_ints:
                for class_wt in class_wts:
                    for intercept_scalinng in intercept_scalinngs:
                        logistic_model = LogisticRegression(C=c,fit_intercept=fit,intercept_scaling=intercept_scalinng,class_weight=class_wt,multi_class='ovr')

                        print(15*"=======")
                        # print("TRAINING CONCATENATE LOGISTIC MODEL")
                        model_concat_fname = 'concate_logistic_model_C_'+str(c)+'_fit_'+str(fit)+'_class_'+str(class_wt)+'_int_'+str(intercept_scalinng)+'.sav'
                        # return _train_model(X_train_concat,X_dev_concat,logistic_model,model_concat_fname,dev_file_list)
                        _train_model(X_train_concat, X_dev_concat, logistic_model, model_concat_fname, dev_file_list)

                        # print("TRAINING SUM LOGISTIC MODEL")
                        # model_fname_sum = 'sum_logistic_model.sav'
                        # _train_model(X_train_sum, X_dev_sum,logistic_model,model_fname_sum,dev_file_list)
                        #
                        # print("TRAINING SUBTRACT LOGISTIC MODEL")
                        # model_fname_subtract = 'subtract_logistic_model.sav'
                        # _train_model(X_train_subtract,X_dev_subtract,logistic_model,model_fname_subtract,dev_file_list)

    def train_cnn(self):

        test_only = False
        model_fname = "word_emb_cnn.h5"

        if test_only:
            event1_train, event2_train, relation_train = [None] * 3

        else:

            train_data = read_vec_data(self.train_data_path, self.num_relations, self.word_context_length,
                                       self.word_vector_size)
            event1_train, event2_train, relation_train = train_data.load_data(load_reverse_data=True)
            # relation_train = np.argmax(relation_train, axis=1)

        dev_data = read_vec_data(self.dev_data_path, self.num_relations, self.word_context_length,
                                 self.word_vector_size)
        event1_dev, event2_dev, relation_dev = dev_data.load_data(load_reverse_data=False)

        relation_dev_index = np.argmax(relation_dev, axis=1)
        dev_file_list = dev_data.get_file_name_list()
        num_relations = 6
        model = models(self.word_context_length, self.word_vector_size)
        train_model = model.get_word_emb_cnn(num_relations)
        if os.path.exists(os.path.join(self.models_path,model_fname)):
            train_model.load_weights(os.path.join(self.models_path, model_fname))
        else:
            train_model.fit([event1_train, event2_train], relation_train, nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE,
                        validation_data=([event1_dev, event2_dev], relation_dev))

        relation_pred = train_model.predict([event1_dev, event2_dev])

        train_model.save_weights(os.path.join(self.models_path,model_fname))
        print("Saved model to file : " + model_fname)
        self.save_report(relation_pred, relation_dev_index, model_fname)
        relation_pred = np.argmax(relation_pred, axis=1)
        self.reuslt_file(dev_file_list, relation_pred)

    def save_report(self,relation_pred, relation_dev_index, m_file_name):

        logging.basicConfig(filename=self.log_file_name, level=logging.DEBUG)
        logging.info(15 * "=====")
        logging.info("Model : " + m_file_name)

        relation_pred = np.argmax(relation_pred, axis=1)
        cls_report = classification_report(relation_dev_index, relation_pred, digits=5)

        # f_score = f1_score(relation_dev_index, relation_pred, average='weighted')
        # self.model_fscore_dict[m_file_name[:-3]] = f_score

        print(cls_report)
        logging.info(str(cls_report))

        logging.info(15 * "=====")

if __name__ == "__main__":

    num_red_relations = 6
    red_rel_vec_files_path = "/home/magnet/onkarp/Data/temporal_relations/processed_data/vector_files/mirza_data"

    tre_sys = tre_logistic_system(red_rel_vec_files_path, num_red_relations)
    tre_sys.train_rel_logistic_model()
    # tre_sys.train_cnn()











