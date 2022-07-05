import os

is_log_debug_messages = True
base_dir = "/srv/storage/magnet@storage1.lille.grid5000.fr/opandit"

data_path =os.path.join(base_dir,"Data/temp_relations")
raw_data_path = os.path.join(data_path,"raw_data")
processed_data_path = os.path.join(data_path,"processed_data")
processed_text_data_path = os.path.join(processed_data_path,"text_files")
processed_vec_data_path = os.path.join(processed_data_path,"vector_files")
log_files_path = os.path.join(base_dir,"Logs/temp_relation_class")
results_path = os.path.join(base_dir,"Results/temp_relation_class")


INTERVAL_RELATIONS = ['AFTER', 'BEFORE', 'DURING', 'DURING_INV', 'SIMULTANEOUS', 'VAGUE']

TEMP_AWARENESS_TOOLKIT_PATH = "/home/magnet/onkarp/Code/temporal_relations/tempeval3_toolkit-master"
TRANSITIVE_RED_EVAL_PATH = "/home/magnet/onkarp/Code/temporal_relations/apetite-0.7/src"
SAVED_MODEL_PATH = "/home/magnet/onkarp/Results/temporal_relations/trained_models"
SAVED_REPORTS = "/home/magnet/onkarp/Results/temporal_relations/reports"
# TML_FILES_PATH = "/home/magnet/onkarp/Data/temporal_relations/raw_data/td_dataset"
TML_FILES_PATH = "/home/magnet/onkarp/Data/temporal_relations/raw_data"
red_rel_vec_files_path = "/home/magnet/onkarp/Data/temporal_relations/processed_data/vector_files/mirza_data"
GOLD_DIR_PATH = "/home/magnet/onkarp/Results/temporal_relations/tml_files/gold_files"
PRED_DIR_PATH = "/home/magnet/onkarp/Results/temporal_relations/tml_files/pred_files"

feat_separator = "$#$#$#$"
NUM_INTERVAL_RELATIONS = 6
NUM_POINT_RELATIONS = 3
WORD_CONTEXT_LENGTH = 9
WORD_VECTOR_SIZE = 300
CHAR_VECTOR_SIZE = 10
VEC_FILES_PATH = "/home/magnet/onkarp/Data/temporal_relations/processed_data/vector_files"
PROCESSED_DATA_PATH = "/home/magnet/onkarp/Data/temporal_relations/processed_data"
RAW_DATA_PATH = "/home/magnet/onkarp/Data/temporal_relations/processed_data/raw_text"
NUM_WORD_FOR_CHAR_EMD = 1

PRED_SCORE_FILE_NAME = "pred_rel_proba_score_pickle_file"

time_bank = "TB"
time_dense = "TD"
vc = "VC"
te3pt = "TE3PT"
aquaint = "AQ"

EXPT_1 = [[time_bank, aquaint, time_dense, vc], [time_dense], [te3pt]]
EXPT_2 = [[time_bank, vc], [time_dense], [te3pt]]
EXPT_3 = [[time_dense], [time_dense], [time_dense]]

event_head_vec_file_extn = "_event_head"
WORD_CONTEXT_NUM = 4
PAD_WORD = "##$$##"

tense_file_extn = "_tense_vec"
pos_file_extn = "_pos_vec"
event_num_file_extn = "_event_num_vec"
line_num_file_extn = "_line_num_vec"

STOPWORDS = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you',
             u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself',
             u'she', u'her', u'hers', u'herself', u'it', u'its', u'itself', u'they', u'them',
             u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that',
             u'these', u'those', u'a', u'an', u'the', u'and', u'but', u'if', u'or',
             u'as', u'of', u'at', u'by', u'for', u'with', u'about', u'against',
             u'between', u'into', u'above', u'below', u'to', u'from', u'up',
             u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'here',
             u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most',
             u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too',
             u'very', u"n't",u',',u'.',u'!',u';']


rel_dict = {'s': 'SIMULTANEOUS', 'i': 'IDENTITY', 'a': 'AFTER', 'v': 'VAGUE', 'ii': 'IS_INCLUDED', 'b': 'BEFORE'}
timeml_file_extn = ".tml"


word2vec_model_path = "/home/opandit/word_embeddings/GoogleNews-vectors-negative300.bin.gz"
fasttext_model_path = "/home/opandit/word_embeddings/wiki-news-300d-1M-subword.vec"