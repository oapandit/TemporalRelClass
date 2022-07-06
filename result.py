import logging
import xml.etree.ElementTree as ET
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import confusion_matrix,precision_score,recall_score,classification_report,f1_score
from constants import *
from utils import *
import TE3_evaluation as te
from apetite.Evaluation import Evaluation
import time
time.strftime("%Y%m%d.%H%M%S")


class EvaluationResult:

    def __init__(self,expt_name=None, gold_dir_path=None, pred_dir_path=None, raw_test_files_path=None,is_test_on_rand_pairs=False):
        expt_name = time.strftime("%Y%m%d.%H%M%S") if expt_name is None else expt_name
        self.gold_dir_path = GOLD_DIR_PATH if gold_dir_path is None else gold_dir_path
        self.pred_dir_path = PRED_DIR_PATH if pred_dir_path is None else pred_dir_path
        self.raw_files_path = TML_FILES_PATH if raw_test_files_path is None else raw_test_files_path
        create_dir(SAVED_REPORTS)
        self.log_file_name = os.path.join(SAVED_REPORTS, expt_name+"_report_.txt")
        self.model_scores_dict = {}
        self.model_fscore_dict_fpath = os.path.join(SAVED_REPORTS,expt_name+"_dir_dict.pkl")
        self.ev_pairs_file_extn = ".ev_pairs" if not is_test_on_rand_pairs else ".rand_ev_pairs"
        self.relation_file_extn = ".rel"
        self.ref_tml_file_path = os.path.join(TML_FILES_PATH, "reference.tml")
        self._initialise_scores()
        self.train_without_vauge = False
        self.bkup_path = self.pred_dir_path + "_bck_up"

    @property
    def train_without_vauge(self):
        return self.train_without_vauge

    @train_without_vauge.setter
    def train_without_vauge(self,val):
        self.train_without_vauge = val

    def _initialise_scores(self):
        self.temp_awareness_score = None
        self.minimal_graph_score = None
        self.direct_fscore = None

    def create_gold_pred_files(self):
        delete_all_data_from_dir(self.gold_dir_path)
        delete_all_data_from_dir(self.pred_dir_path)


    def backup_pred_dir(self):
        self.bkup_path= self.pred_dir_path+"_bck_up"
        shutil.rmtree(self.bkup_path)
        shutil.copytree(self.pred_dir_path, self.bkup_path)

    def test_event_pairs_pred_equal(self,file_list,num_pred):

        num_ev_pairs =0
        num_rel =0
        for f in file_list:
            # print f
            ev_pairs_file = f[:-4]+self.ev_pairs_file_extn
            rel_file = f[:-4]+self.relation_file_extn
            with open(search_single_file_in_dir(PROCESSED_DATA_PATH,ev_pairs_file), "r") as f:
                ev_pairs = f.readlines()
            num_ev_pairs +=len(ev_pairs)
            print("For event pairs file {} , number of pairs are {}".format(f,len(ev_pairs)))
            with open(search_single_file_in_dir(PROCESSED_DATA_PATH,rel_file), "r") as f:
                rels = f.readlines()
            num_rel +=len(rels)


        print("number of predictions",num_pred)
        print("number of pairs",num_ev_pairs)
        print("number of relations",num_rel)


    def create_pred_file(self, relation_pred, file_list):

        if not isinstance(file_list, list):
            file_list = [file_list]

        if relation_pred is None:
            relation_pred = load_from_pickle_file(os.path.join(SAVED_MODEL_PATH, PRED_SCORE_FILE_NAME))
        # else:
        print(("predicted relations data shape : {0}".format(relation_pred.shape)))
        if len(relation_pred.shape)==2:
            relation_pred = np.argmax(relation_pred, axis=1)
        print(("predicted relations data shape after doing argmax : {0}".format(relation_pred.shape)))
            # self.create_proba_score_file(relation_pred)
        print(file_list)
        counter = 0
        self.test_event_pairs_pred_equal(file_list,relation_pred.shape[0])
        for f in file_list:
            # print(f)
            if not os.path.isfile(os.path.join(self.gold_dir_path,f)):
                self.copy_gold_annotated_files(f)
            _xmldoc = ET.parse(os.path.join(self.gold_dir_path,f))
            f_new_path = os.path.join(self.pred_dir_path,f)
            tlinks = _xmldoc.findall('TLINK')
            root = _xmldoc.getroot()
            ev_pairs_file = f[:-4]+self.ev_pairs_file_extn
            rel_file = f[:-4]+self.relation_file_extn
            # with open(os.path.join(VEC_FILES_PATH+"/dev_data", ev_pairs_file), "r") as f:
            #     ev_pairs = f.readlines()

            with open(search_single_file_in_dir(PROCESSED_DATA_PATH,ev_pairs_file), "r") as f:
                ev_pairs = f.readlines()

            rel_ev_pairs_file_path = search_single_file_in_dir(PROCESSED_DATA_PATH,rel_file)
            rel_of_ev_pairs = None
            if rel_ev_pairs_file_path is not None:
                with open(rel_ev_pairs_file_path, "r") as f:
                    rel_of_ev_pairs = f.readlines()

            lid = 500

            # Deleting all gold pairs relation
            for tlink in tlinks:
                root.remove(tlink)

            for ind,ev_pair in enumerate(ev_pairs):
                if rel_of_ev_pairs is not None and self.train_without_vauge:
                    if rel_of_ev_pairs[ind].strip().upper()=="VAGUE":
                        continue
                ev_pair = ev_pair.strip()
                ev1_eiid,ev2_eiid = ev_pair.split(feat_separator)
                pred_relation = INTERVAL_RELATIONS[relation_pred[counter]]

                if pred_relation != INTERVAL_RELATIONS[5]:# checking if the relation is vague
                    new_tlink = ET.Element("TLINK")
                    new_tlink.set("lid", "l" + str(lid))
                    new_tlink.set("relType", pred_relation)
                    new_tlink.set("eventInstanceID", ev1_eiid)
                    new_tlink.set("relatedToEventInstance", ev2_eiid)
                    new_tlink.tail = "\n"
                    root.append(new_tlink)
                    lid+=1
                counter += 1

            # for tlink in tlinks:
            #     if counter == relation_pred.shape[0]:
            #         break
            #     if "eventInstanceID" in tlink.attrib and "relatedToEventInstance" in tlink.attrib:
            #         pred_relation = INTERVAL_RELATIONS[relation_pred[counter]]
            #         if pred_relation == INTERVAL_RELATIONS[5]: # checking if the relation is vague
            #             root.remove(tlink)
            #         else:
            #             tlink.set('relType', pred_relation)
            #         counter+=1
            _xmldoc.write(f_new_path)

        if counter!=relation_pred.shape[0]:
            print("ERROR not equal")
        print(counter)

    def copy_gold_annotated_files(self, file_list):
        if not isinstance(file_list, list):
            file_list = [file_list]
        newline_char = "\n"

        for f in file_list:

            ref_tml_cont = ET.parse(self.ref_tml_file_path)
            ref_root = ref_tml_cont.getroot()

            # print(f)
            tml_file_name = f[:-10]+".tml" if f.endswith("_TE3PT.tml") else f
            src = search_single_file_in_dir(self.raw_files_path, tml_file_name)
            # print(src)
            # src = os.path.join(self.raw_files_path, f)
            dst = os.path.join(self.gold_dir_path, f)
            # copyfile(src, dst)
            # original files contain information about relations between timex but we are not predicting those relations
            # at all,so deleting those links from original files  for fair comparison
            _xmldoc = ET.parse(src)
            tlinks = _xmldoc.findall('TLINK')
            root = _xmldoc.getroot()
            instances = _xmldoc.findall('MAKEINSTANCE')

            #Get all event ids.
            event_id_list =[]
            for instance in instances:
                if "eventID" in instance.attrib:
                    event_id_list.append(instance.attrib["eventID"])

            # print(event_id_list)
            event_id_list = list(set(event_id_list))
            # print(event_id_list)

            #Create TEXT element to add this event ids
            text_elem = ET.Element("TEXT")
            text_elem.tail = newline_char
            text_elem.text = "\n \n \nThis is really dummy"

            for event_id in event_id_list:
                event_elem = ET.SubElement(text_elem,"EVENT")
                event_elem.set("class", "OCCURRENCE")
                event_elem.set("eid", event_id)
                event_elem.text = "DUMMY"
                event_elem.tail = "."+newline_char+"This is really dummy"
            event_elem.tail = ".\n \n \n "
            ref_root.append(text_elem)

            # Append instances of events
            for instance in instances:
                ref_root.append(instance)

            # Append tlinks
            for tlink in tlinks:

                if "signalID" in tlink.attrib:
                    del tlink.attrib["signalID"]

                if "timeID" in tlink.attrib or "relatedToTime" in tlink.attrib or  tlink.attrib['relType'] == 'VAGUE':
                    root.remove(tlink)
                else:
                    ref_root.append(tlink)
            # tlinks = _xmldoc.findall('TLINK')

            # _xmldoc.write(dst)
            ref_tml_cont.write(dst)

    def evaluate_with_temporal_awareness(self,y_pred=None, file_list=None):
        score = None
        if y_pred is not None:
            self.create_gold_pred_files()
            self.copy_gold_annotated_files(file_list)
            self.create_pred_file(y_pred,file_list)
        try:
            score= self.execute_temporal_awareness(self.gold_dir_path, self.pred_dir_path)
        except TimedOutExc as e:
            print("Its taking too much time ..... Halting temporal awareness metric evaluation.")
        return score


    def evaluate_with_minimal_graph(self,y_pred=None, file_list=None):
        score = None
        if y_pred is not None:
            self.create_gold_pred_files()
            self.copy_gold_annotated_files(file_list)
            self.create_pred_file(y_pred,file_list)
        try:
            score = self.execute_minimal_graph_evaluation(self.gold_dir_path, self.pred_dir_path)
        except TimedOutExc as e:
            print("Its taking too much time ..... Halting transitive reduction metric evaluation.")
        return score


    def round_of_score(self,score_list):
        scores = []
        for score in score_list:
            if score<1:
                score = 100*score
            scores.append(round(score,2))
        return scores

    @deadline(600)#wait for 10 mins; otherwise quit
    def execute_temporal_awareness(self, gold_dir_path, pred_dir_path):
        print(("####" * 30))
        print("Evaluating system with temporal awareness metric")
        self.temp_awareness_score = te.te3_evaluate(gold_dir_path, pred_dir_path, debug_val=0)
        self.temp_awareness_score = self.round_of_score(self.temp_awareness_score)
        return self.temp_awareness_score

    @deadline(600)#wait for 10 mins; otherwise quit
    def execute_minimal_graph_evaluation(self,gold_dir_path, pred_dir_path):
        print(("####" * 30))
        print("Testing with Transitive reduction metric")
        print(("####" * 30))
        measures = ["tr_recall", "tr_prec"]
        evaluation = Evaluation(gold_dir_path, pred_dir_path, ascii_ref=False)
        evaluation.compute(measures=measures)
        detail_results,self.minimal_graph_score = evaluation.report(verbose=True)
        print(detail_results)
        self.minimal_graph_score = self.round_of_score(self.minimal_graph_score)
        return self.minimal_graph_score


    def create_proba_score_file(self, rel_pred_scores, f_name=PRED_SCORE_FILE_NAME):
        print("Saving probability scores to file")
        save_as_pickle_file(rel_pred_scores,os.path.join(SAVED_MODEL_PATH, f_name))

    def is_prob_score_file_exist(self,f_name=PRED_SCORE_FILE_NAME):
        return os.path.isfile(os.path.join(SAVED_MODEL_PATH, f_name))

    def read_prob_score_file(self,f_name=PRED_SCORE_FILE_NAME):
        return load_from_pickle_file(os.path.join(SAVED_MODEL_PATH, f_name))

    def mcnemar_stats(self,relation_gold, relation_pred):

        relation_gold = np.argmax(relation_gold, axis=1)
        relation_pred = np.argmax(relation_pred, axis=1)

        cf_table_1 = confusion_matrix(relation_gold, relation_pred)
        print(cf_table_1)
        # mn_score = mcnemar(x=cf_table,y= cf_table_1,exact=False, correction=True)
        mn_score = mcnemar(cf_table_1, exact=False, correction=True)
        print(mn_score)
        stat = sum(mn_score[0])
        print(stat)
        self.mcnemar_stat_list.append(stat)
        print((mn_score.statistic))
        print((mn_score.pvalue))


    def evaluate_direct(self,relation_gold, relation_pred,is_print_report=False):
        if len(relation_gold.shape) == 2:
            relation_gold = np.argmax(relation_gold, axis=1)
        if len(relation_pred.shape) == 2:
            relation_pred = np.argmax(relation_pred, axis=1)

        if relation_gold.shape[0]!= relation_pred.shape[0]:
            print("Number of samples in gold data and predicted data do not match....returning 0 score.")
            self.direct_fscore = [0, 0, 0]
            return self.direct_fscore

        else:
            cls_report = classification_report(relation_gold, relation_pred, digits=5)
            if is_print_report:
                print(cls_report)
            f = f1_score(relation_gold, relation_pred, average='weighted')
            p = precision_score(relation_gold, relation_pred, average='weighted')
            r = recall_score(relation_gold, relation_pred, average='weighted')
            self.direct_fscore = self.round_of_score([p,r,f])
        return self.direct_fscore


    def log_scores_in_file(self, m_file_name):

        _dict ={}

        score_names = ["P","R","F"]

        logging.basicConfig(filename=self.log_file_name, level=logging.DEBUG)
        logging.info(15 * "=====")
        logging.info("Model : " + m_file_name)
        logging.info("Scores : "+"\t".join(score_names))
        if self.temp_awareness_score is not None:
            _dict["temp_awareness_score"] = self.temp_awareness_score
            # scores_ = "\t".join(self.temp_awareness_score)
            if len(self.temp_awareness_score)>=3:
                scores_ = "{0}\t{1}\t{2}".format(*self.temp_awareness_score)
                logging.info("temp_awareness_score : "+scores_)

        if self.minimal_graph_score is not None:
            _dict["minimal_graph_score"] = self.minimal_graph_score
            # scores_ = "\t".join(self.minimal_graph_score)
            if len(self.minimal_graph_score)>=3:
                scores_ = "{0}\t{1}\t{2}".format(*self.minimal_graph_score)
                logging.info("minimal_graph_score"+scores_)

        if self.direct_fscore is not None:
            _dict["direct_fscore"] = self.direct_fscore
            # scores_ = "\t".join(self.direct_fscore)
            if len(self.direct_fscore)>=3:
                scores_ = "{0}\t{1}\t{2}".format(*self.direct_fscore)
                logging.info("direct_fscore :  "+scores_)

        self.model_scores_dict[m_file_name[:-3]] = _dict
        logging.info(15 * "=====")
        self._initialise_scores()


    def compare_gold_pair_pred_glob_cohere_pred_results(self):

        compare_all = False
        compare_only_pair = False
        compare_only_glob = False

        gold_files = get_list_of_files_in_dir(self.gold_dir_path)
        pair_pred_files = get_list_of_files_in_dir(self.bkup_path)
        glob_coh_pred_files = get_list_of_files_in_dir(self.pred_dir_path)

        if len(gold_files) == len(pair_pred_files) == len(glob_coh_pred_files):
            compare_all = True
        elif len(gold_files) == len(glob_coh_pred_files):
            compare_only_glob = True
        elif len(gold_files) == len(pair_pred_files):
            compare_only_pair = True


        for file in gold_files:
            gold_xmldoc = ET.parse(os.path.join(self.gold_dir_path,file))

            gold_tlinks = gold_xmldoc.findall('TLINK')

            glob_coh_tlinks = None
            pair_pred_tlinks = None

            if compare_all:
                glob_coh_xmldoc = ET.parse(os.path.join(self.pred_dir_path,file))
                glob_coh_tlinks = glob_coh_xmldoc.findall('TLINK')

                pair_pred_xmldoc = ET.parse(os.path.join(self.bkup_path,file))
                pair_pred_tlinks = pair_pred_xmldoc.findall('TLINK')

            elif compare_only_glob:
                glob_coh_xmldoc = ET.parse(os.path.join(self.pred_dir_path, file))
                glob_coh_tlinks = glob_coh_xmldoc.findall('TLINK')
            elif compare_only_pair:
                pair_pred_xmldoc = ET.parse(os.path.join(self.bkup_path, file))
                pair_pred_tlinks = pair_pred_xmldoc.findall('TLINK')


            for ind,tlink in enumerate(gold_tlinks):
                ev1 = tlink.attrib["eventInstanceID"]
                ev2 = tlink.attrib["relatedToEventInstance"]
                gold_rel = tlink.attrib["relType"]

                print("Gold relation for pair {} and {} is {}. ".format(ev1,ev2,gold_rel))

                if pair_pred_tlinks is not None:
                    _ev1 = pair_pred_tlinks[ind]["eventInstanceID"]
                    _ev2 = pair_pred_tlinks[ind]["relatedToEventInstance"]
                    _pair_pred_rel = pair_pred_tlinks[ind]["relType"]

                    if _ev1 == ev1 and _ev2 == ev2:
                        print("Predicted pairwise relation for pair {} and {} is {}. ".format(_ev1, _ev2,
                                                                                                       _pair_pred_rel))

                if glob_coh_tlinks is not None:
                    _ev1 = glob_coh_tlinks[ind]["eventInstanceID"]
                    _ev2 = glob_coh_tlinks[ind]["relatedToEventInstance"]
                    _glob_rel = glob_coh_tlinks[ind]["relType"]

                    if _ev1 == ev1 and _ev2 == ev2:
                        print("Predicted globally coherent relation for pair {} and {} is {}. ".format(_ev1, _ev2,_glob_rel))








        pass

    def save_score_dicts(self):
        save_as_pickle_file(self.model_scores_dict, self.model_fscore_dict_fpath)


if __name__ == '__main__':
    gold_dir_path = "/home/magnet/onkarp/Data/temporal_relations/raw_data/te3-platinum"
    f_list = ['APW19980227.0487_TD.tml', 'CNN19980223.1130.0960_TD.tml', 'NYT19980212.0019_TD.tml',
              'PRI19980216.2000.0170_TD.tml', 'ed980111.1130.0089_TD.tml']
    er =  EvaluationResult()
    score = er.evaluate_with_temporal_awareness()
    print(score)
    score = er.evaluate_with_minimal_graph()
    print(score)
    # er.test_event_pairs_pred_equal(f_list,675)
    # er.execute_minimal_graph_evaluation()
    # er.execute_temporal_awareness(gold_dir_path,gold_dir_path)
    # er.log_scores_in_file("")