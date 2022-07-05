import os
import pickle
from utils import *
from constants import *
import xml.etree.ElementTree as ET
from result import EvaluationResult
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import shuffle
import random
import shutil

EVAL_EXPTS_PATH = "/home/magnet/onkarp/Results/temporal_relations/evaluation_expts"

class temporal_awareness_expt():

    def __init__(self,gold_dir=None,pred_dir=None):
        # self.test_files_pat
        self.get_list_of_tml_files()
        self.raw_text_path = TML_FILES_PATH
        self.gold_dir = GOLD_DIR_PATH if gold_dir is None else gold_dir
        self.pred_dir = PRED_DIR_PATH if pred_dir is None else pred_dir
        self.res = EvaluationResult()


    def get_list_of_tml_files(self):
        self.tml_files_list = ["CNN_20130322_1003.tml"]
        return self.tml_files_list

    def get_random_tm_file(self):
        file_list = search_all_file_in_dir(self.raw_text_path,"*.tml")
        # file_list = [search_single_file_in_dir(self.raw_text_path,"APW19980818.0515.tml")]
        return random.choice(file_list)



    def delete_previous_files(self):
        delete_all_data_from_dir(os.path.dirname(self.gold_dir))
        delete_all_data_from_dir(self.gold_dir)
        delete_all_data_from_dir(self.pred_dir)

    def copy_files_to_gold_location(self,tml_files_list=None):
        self.delete_previous_files()
        tml_files_list = self.tml_files_list if tml_files_list is None else tml_files_list
        for tml_file in tml_files_list :
            if not os.path.isfile(tml_file):
                tml_file_source_path = search_single_file_in_dir(self.raw_text_path,tml_file)
                tml_file_dest_path = os.path.join(self.gold_dir, tml_file)
            else:
                tml_file_source_path = tml_file
                tml_file_dest_path = os.path.join(self.gold_dir, os.path.basename(tml_file))
            # print "source path",tml_file_source_path
            # print "destination path",tml_file_dest_path
            shutil.copy2(tml_file_source_path,tml_file_dest_path)

    def copy_files_to_pred_location(self,percent=20,is_shuffle_pairs=False):

        tml_file_list_at_gold_loc = get_list_of_files_in_dir(self.gold_dir)

        for tml_file in tml_file_list_at_gold_loc:
            gold_tml_path = os.path.join(self.gold_dir,tml_file)
            pred_tml_path = os.path.join(self.pred_dir,tml_file)

            gold_tml_cont = ET.parse(gold_tml_path)
            gold_tml_root = gold_tml_cont.getroot()

            tlinks = gold_tml_cont.findall('TLINK')
            num_tlinks = len(tlinks)
            num_tlinks_to_be_deleted = int(num_tlinks*percent/100)

            # if self.num_tlinks_to_be_deleted is None:
            #     self.num_tlinks_to_be_deleted = num_tlinks_to_be_deleted
            #     self.tlinks_to_be_deleted = []
            #     for tlink in tlinks[0:num_tlinks_to_be_deleted]:
            #         self.tlinks_to_be_deleted.append(tlink.attrib['lid'])
            #
            #     self.tlinks_to_be_deleted.sort()
            #
            #     # print self.tlinks_to_be_deleted
            #     # print self.num_tlinks_to_be_deleted
            # else:
            #     if self.num_tlinks_to_be_deleted != num_tlinks_to_be_deleted:
            #         print self.num_tlinks_to_be_deleted
            #     # print num_tlinks_to_be_deleted
            #
            #     _tlinks_to_be_deleted = []
            #     for tlink in tlinks[0:num_tlinks_to_be_deleted]:
            #         _tlinks_to_be_deleted.append(tlink.attrib['lid'])
            #
            #     _tlinks_to_be_deleted.sort()
            #     if self.tlinks_to_be_deleted!= _tlinks_to_be_deleted:
            #         print self.tlinks_to_be_deleted
            #     # print _tlinks_to_be_deleted


            # _tlinks_to_be_deleted = []
            # for tlink in tlinks[0:num_tlinks_to_be_deleted]:
            #     _tlinks_to_be_deleted.append(tlink.attrib['lid'])
            #
            # _tlinks_to_be_deleted.sort()
            # print _tlinks_to_be_deleted

            for tlink in tlinks:
                gold_tml_root.remove(tlink)


            # print "Number of links to be deleted",num_tlinks_to_be_deleted

            tlinks = tlinks[num_tlinks_to_be_deleted:]

            if is_shuffle_pairs:
                random.shuffle(tlinks)

            for tlink in tlinks:
                # print tlink.attrib['lid']
                gold_tml_root.append(tlink)

            gold_tml_cont.write(pred_tml_path)

    def save_pred_file(self):

        tml_file_list_at_gold_loc = get_list_of_files_in_dir(self.pred_dir)
        for tml_file in tml_file_list_at_gold_loc:
            pred_tml_path = os.path.join(self.pred_dir, tml_file)
            save_tml_path = os.path.join(os.path.dirname(self.pred_dir), tml_file)
            shutil.copy2(pred_tml_path,save_tml_path)


    def delete_single_tlink_from_pred(self):

        tml_file_list_at_gold_loc = get_list_of_files_in_dir(os.path.dirname(self.pred_dir))

        for tml_file in tml_file_list_at_gold_loc:
            pred_tml_path = os.path.join(self.pred_dir,tml_file)

            saved_pred_tml_path = os.path.join(os.path.dirname(self.pred_dir), tml_file)

            tml_cont = ET.parse(saved_pred_tml_path)
            tml_root = tml_cont.getroot()

            tlinks = tml_cont.findall('TLINK')

            deleted_lid = "No tlink"
            if len(tlinks)>0:
                for tlink in tlinks:
                    tml_root.remove(tlink)

                random.shuffle(tlinks)
                # for tlink in tlinks:
                #     print tlink.attrib['lid']
                deleted_lid = tlinks[0].attrib['lid']
                # print "Deleted id",deleted_lid
                tlinks = tlinks[1:]

                for tlink in tlinks:
                    # print tlink.attrib['lid']
                    tml_root.append(tlink)

                tml_cont.write(pred_tml_path)

        return deleted_lid






    def delete_event_pairs_monotonously_expt(self):
        self.copy_files_to_gold_location()

        reduced_percent_relation_list = range(0,101,5)
        precision_list = []
        recall_list = []
        fscore_list = []

        for reduced_percent_relation in reduced_percent_relation_list:
            self.copy_files_to_pred_location(percent=reduced_percent_relation,is_shuffle_pairs=False)
            p,r,f = self.res.execute_temporal_awareness(self.gold_dir,self.pred_dir)
            precision_list.append(p)
            recall_list.append(r)
            fscore_list.append(f)

        plt.plot(reduced_percent_relation_list, precision_list,label = "Precision")
        plt.plot(reduced_percent_relation_list, recall_list,label = "Recall")
        plt.plot(reduced_percent_relation_list, fscore_list,label = "Fscore")

        plt.legend(loc='lower left')

        plt.xlabel("% reduction in relations")
        plt.ylabel("Scores")
        plt.savefig('myfig.pdf')


    def expt_to_see_inconsistancy(self):

        def is_all_scores_not_same(iterator):
            'method to check if all elements in list are same.'
            return len(set(iterator)) > 1

        random_tml_file = self.get_random_tm_file()
        print "File for the experiment : ",random_tml_file
        self.copy_files_to_gold_location([random_tml_file])

        # reduced_percent_relation_list = range(5, 101, 5)
        # reduced_percent_relation_list = range(10, 20, 5)
        reduced_percent_relation_list = [random.randint(0,99)]

        for reduced_percent_relation in reduced_percent_relation_list:
            self.num_tlinks_to_be_deleted = None
            self.tlinks_to_be_deleted = None
            precision_list = []
            recall_list = []
            fscore_list = []
            plot_fname = 'expt_inconsi_{}.pdf'.format(reduced_percent_relation)
            recall_fname = '{}_{}_recall.pkl'.format(os.path.basename(random_tml_file),reduced_percent_relation)
            recall_f_path = os.path.join(EVAL_EXPTS_PATH,"inconsistency_expt",recall_fname)
            plot_f_path = os.path.join(EVAL_EXPTS_PATH,"inconsistency_expt",plot_fname)
            print "Deleting {} percent of tlinks".format(reduced_percent_relation)
            repeat_expt = 100
            # num_list = range(repeat_expt)
            num_list =0
            no_change =0
            # for expt in num_list:
            while True:
                if no_change ==repeat_expt:
                    print "No change for last {} runs so breaking the loop.".format(repeat_expt)
                    break
                self.copy_files_to_pred_location(percent=reduced_percent_relation, is_shuffle_pairs=True)
                p, r, f = self.res.execute_temporal_awareness(self.gold_dir, self.pred_dir)
                precision_list.append(p)
                recall_list.append(r)
                fscore_list.append(f)
                if len(recall_list) >2:
                    if recall_list[-1] == recall_list[-2]:
                        no_change+=1
                        print "No change."
                    else:
                        no_change =0
                        print "Change in score."
                num_list+=1
                print 40 * "+++"

            num_list = range(num_list)

            if is_all_scores_not_same(precision_list) or is_all_scores_not_same(recall_list) or is_all_scores_not_same(fscore_list):
                print "Variation in scores; plotting scores"
                plt.figure()
                # plt.plot(num_list, precision_list, label="Precision")
                # plt.plot(num_list, recall_list, label="Recall")
                plt.plot(num_list, recall_list)
                plt.plot(num_list, recall_list,'r^')
                # plt.plot(num_list, fscore_list, label="Fscore")

                # plt.legend(loc='lower left')

                plt.xlabel("Expt number")
                plt.ylabel("Scores")
                plt.title('Expt for file {} and percentage reduction {}'.format(os.path.basename(random_tml_file),reduced_percent_relation))
                plt.savefig(plot_f_path)
                plt.close()

                save_as_pickle_file(np.array(recall_list),recall_f_path)


            else:
                print "Scores are same; no need to plot."
            print 40*"==="



    def expt_to_see_same_result_after_deleting_tlink(self):

        random_tml_file = self.get_random_tm_file()
        print "File for the experiment : ",random_tml_file
        self.copy_files_to_gold_location([random_tml_file])

        reduced_percent_relation = random.randint(0,99)

        self.copy_files_to_pred_location(percent=reduced_percent_relation, is_shuffle_pairs=False)
        p, r, f = self.res.execute_temporal_awareness(self.gold_dir, self.pred_dir)
        self.save_pred_file()
        print 40 * "+++"
        repeat_expt = 100
        no_change = 0
        while True:
            if no_change ==repeat_expt:
                print "Result has always changed in last {} runs so breaking the loop.".format(repeat_expt)
                break
            deleted_lid =self.delete_single_tlink_from_pred()
            _p, _r, _f = self.res.execute_temporal_awareness(self.gold_dir, self.pred_dir)

            if int(r)!=0 and _r==r:
                print "Found file where deleting tlink doesn't reduce the result {}".format(os.path.basename(random_tml_file))
                zip_path = os.path.join(EVAL_EXPTS_PATH, "remove_expt", deleted_lid)
                shutil.make_archive(zip_path, 'zip', os.path.dirname(self.gold_dir))
                break
            else:
                no_change+=1
            print 20 * "=="



if __name__ == '__main__':
    tae = temporal_awareness_expt()
    # tae.delete_event_pairs_monotonously_expt()
    # tae.expt_to_see_inconsistancy()
    tae.expt_to_see_same_result_after_deleting_tlink()







