import os
import pickle
import numpy as np
from numpy import linalg
import pulp
# import xml.etree.ElementTree as ET
from lxml import etree as ET
from shutil import copy2
import TE3_evaluation as te
from constants import *
from utils import *
import logging
SAVED_MODEL_PATH = "/home/magnet/onkarp/Results/temporal_relations"
f_name = "rel_proba_score"
import random

processed_but_raw_data_path = "/home/magnet/onkarp/Data/temporal_relations/processed_data/raw_text"

# TML_FILES_PATH ="/home/magnet/onkarp/Data/temporal_relations/raw_data/te3-platinum"
RAW_TEXT_FILES = "/home/magnet/onkarp/Data/temporal_relations/raw_data/td_dataset"
toolkit_path = "/home/magnet/onkarp/Code/temporal_relations/tempeval3_toolkit-master"
from constants import *
from apetite.Evaluation import Evaluation
import time
# from tre_system import tre_system
from result import EvaluationResult
from numpy import linalg as LA



class maintain_consistency(object):
    start = 0
    end = 0

    def __init__(self):
        self.num_interval_relations = NUM_INTERVAL_RELATIONS
        self.num_base_point_relations = NUM_POINT_RELATIONS
        # self.test_path = test_path
        # self.num_base_point_relations = 5
        self.ev_pair_file_extn = ".ev_pairs"
        self.num_point_relations = self.num_base_point_relations + 2
        self.ev_pairs = None
        self.result = EvaluationResult()
        pass


    # @property
    # def ev_pairs(self):
    #     return self.ev_pairs
    #
    # @ev_pairs.setter
    # def ev_pairs(self,val):
    #     self.ev_pairs = val


    def get_list_of_files(self, dir_path, file_extension=".ev_pairs"):

        filelist = [name[:-4] for name in os.listdir(dir_path) if
                    name.endswith(".ev_pairs_vec") and not os.path.isdir(os.path.join(dir_path, name))]
        for file in filelist:
            copy2(os.path.join(processed_but_raw_data_path, file), os.path.join(dir_path, file))
        filelist.sort()

        # filelist = [name for name in os.listdir(dir_path) if name.endswith(file_extension) and  not os.path.isdir(os.path.join(dir_path, name))]
        # filelist.sort()
        return filelist

    def get_start_end_index(self, event_pairs):
        maintain_consistency.end = maintain_consistency.end + len(event_pairs)
        return maintain_consistency.start, maintain_consistency.end

    def get_events_from_event_pairs(self,event_pairs):

        events = []
        for ind, line in enumerate(event_pairs):
            # print("-----" * 10)
            e1,e2 = line.strip().split(feat_separator)
            events.append(e1)
            events.append(e2)
        events = sorted(set(events))

        return events

    def read_interval_relations_events_and_probability_score(self,filelist,probability_score=None):
        # filelist = self.get_list_of_files(self.test_path)
        if probability_score is None:
            probability_score = load_from_pickle_file(os.path.join(SAVED_MODEL_PATH, PRED_SCORE_FILE_NAME))

        print("probability score dimentions :", probability_score.shape)

        # probability_score = probability_score.reshape(-1,self.num_relations)
        for file in filelist:
            print("####" * 20)
            print("FILE : ", file)
            with open(os.path.join(processed_but_raw_data_path, file[:-4]+self.ev_pair_file_extn), "r") as f:
                event_pairs = f.readlines()
            start, end = self.get_start_end_index(event_pairs)
            print("START : ", start)
            print("END : ", end)
            # events, rel_pred = self.optimize_exact_interval_lim_relations(event_pairs, probability_score[start:end, ])
            if file == "ed980111.1130.0089_TD.tml":
                start_time = time.time()
                # self.optimize_exact_point_relations(event_pairs,probability_score[start:end, ])
                events, rel_pred = self.optimize_exact_interval_lim_relations(event_pairs, probability_score[start:end, ])
                print("--- {0} seconds ---".format(time.time() - start_time))
            # self.generate_final_prediction_file(file,events, rel_pred)
            maintain_consistency.start = maintain_consistency.end

        # dir = "dir1"
        # res_path = os.path.join(SAVED_MODEL_PATH, dir)
        # gold_dir_path = os.path.join(res_path, "gold_files")
        # pred_dir_path = os.path.join(res_path, "pred_files")
        #
        # self.get_final_result_report(gold_dir_path, pred_dir_path)
        # maintain_consistency.start =0
        # maintain_consistency.end = 0

    def read_end_point_relations_events_and_probability_score(self,filelist,probability_score=None):
        if probability_score is None:
            probability_score = load_from_pickle_file(os.path.join(SAVED_MODEL_PATH, PRED_SCORE_FILE_NAME))

        print("probability score dimentions :", probability_score.shape)

        # for i in range(len(probability_score)):
        #     print("Point relation data " + str(i) + " : shape : " + str(probability_score[i].shape))
        # print("probability score dimentions :", probability_score.shape)
        probability_score_np_array = np.array(probability_score)
        print(probability_score_np_array.shape)
        # probability_score = probability_score.reshape(-1,self.num_relations)
        self.result.backup_pred_dir()
        self.result.create_gold_pred_files()
        for file in filelist:
            print("####" * 20)
            print("FILE : ", file)
            with open(os.path.join(processed_but_raw_data_path, file[:-4]+".ev_pairs"), "r") as f:
                event_pairs = f.readlines()
            start, end = self.get_start_end_index(event_pairs)
            print("START : ", start)
            print("END : ", end)
            if self.result.is_prob_score_file_exist(file[:-4]+"_glob_pred.pkl"):
                start_time = time.time()
                events, rel_pred = self.optimize_lagrangian_relaxed_point_relations(event_pairs, probability_score[start:end,])
                print("--- {0} seconds ---".format(time.time() - start_time))
                self.result.create_proba_score_file(rel_pred,file[:-4]+"_glob_pred.pkl")
            else:
                sys.exit("Files exists...exiting")
                events = self.get_events_from_event_pairs(event_pairs)
                rel_pred = self.result.read_prob_score_file(file[:-4]+"_glob_pred.pkl")
            more_than_one_base_rel_list_of_end_points, trans_constr_vio_list_of_end_points, inverse_constr_vio_list_of_end_points, opposite_relations_list_of_end_points = self.check_constraints_on_end_points(rel_pred)
            rel_pred = self.heauristics_to_correct_point_relations(rel_pred,more_than_one_base_rel_list_of_end_points,trans_constr_vio_list_of_end_points,inverse_constr_vio_list_of_end_points,opposite_relations_list_of_end_points)

            rel_pred = self.convert_point_to_interval_relations(rel_pred)
            rel_pred = self.keep_only_gold_event_pair_relations(rel_pred,event_pairs,events)
            self.result.create_pred_file(rel_pred,file)
            maintain_consistency.start = maintain_consistency.end
            sys.exit("exitting after one iteration")

        self.result.execute_temporal_awareness(GOLD_DIR_PATH,PRED_DIR_PATH)

    def optimize_exact_interval_lim_relations(self, event_pairs, probability_score):

        print("probability score dimentions :", probability_score.shape)

        a = 0
        b = 1
        d = 2
        di = 3
        id = 4
        v = 5

        def __get_proba_score(i, j):

            proba_score = [0] * self.num_interval_relations
            if i==j:
                proba_score[id]=1
                return proba_score

            e1 = events[i]
            e2 = events[j]

            ev_pair = e1+feat_separator+e2
            if ev_pairs_rel_dict.get(ev_pair,None) is None:
                return proba_score
            else:
                return ev_pairs_rel_dict.get(ev_pair)

        def __get_reverse_score(prob_score):
            rev_prob_score = [0] * self.num_interval_relations
            rev_prob_score[a] = prob_score[b]
            rev_prob_score[b] = prob_score[a]
            rev_prob_score[d] = prob_score[di]
            rev_prob_score[di] = prob_score[d]
            rev_prob_score[id] = prob_score[id]
            rev_prob_score[v] = prob_score[v]
            return rev_prob_score

        events = []


        ev_pairs_rel_dict = {}
        print("Reading from probability score matrix...")
        for ind, line in enumerate(event_pairs):
            # print("-----" * 10)
            line = line.strip().split(feat_separator)
            e1 = line[0]
            e2 = line[1]
            p_score = probability_score[ind, :].tolist()
            rev_p_score = __get_reverse_score(p_score)

            pair = e1+feat_separator+e2
            rev_pair = e2+feat_separator+e1

            ev_pairs_rel_dict[pair] = p_score
            ev_pairs_rel_dict[rev_pair] = rev_p_score

            # print("For event pair {0} and {1}, probability score is {2}".format(e1,e2,p_score))
            # print("For event reverse pair {0} and {1}, probability score is {2}".format(e2,e1,rev_p_score))

            events.append(e1)
            events.append(e2)

        events = sorted(set(events))
        num_events = len(events)

        print("Number of events", num_events)
        # defining indicator variable which can take value 0 or 1
        I = pulp.LpVariable.dicts('I', (range(num_events), range(num_events), range(self.num_interval_relations)), lowBound=0,
                                  upBound=1,
                                  cat=pulp.LpInteger)

        # probability score
        p = []
        for i in range(num_events):
            p.append([])
            for j in range(num_events):
                p[i].append([])
                p[i][j] = __get_proba_score(i, j)

        ## for debugging purpose
        counter = 0
        for i in range(num_events):
            for j in range(num_events):
                counter+=1
                if __get_proba_score(i,j) != p[i][j]:
                    print("-----"*10)
                    print("Event pair : {0} and {1} ".format(events[i],events[j]))
                    print("Probability score is : {0}".format(__get_proba_score(i,j)))
                    print("Probability score entered is : {0}".format(p[i][j]))
        # print(counter)
        # print("#$#$"*20)
        print("Total number of variables : ", num_events * num_events * self.num_interval_relations)
        model = pulp.LpProblem("Maximizing score", pulp.LpMaximize)

        # Objective Function
        model += (
            pulp.lpSum([p[i][j][k] * I[i][j][k] for k in range(self.num_interval_relations) for j in range(num_events) for i in
                        range(num_events)])
        )

        # constraints

        for i in range(num_events):
            for j in range(num_events):
                model += (pulp.lpSum([I[i][j][k] for k in range(self.num_interval_relations)]) == 1)
                if i != j:
                    model += I[i][j][a] == I[j][i][b]
                    model += I[i][j][d] == I[j][i][di]
                    model += I[i][j][id] == I[j][i][id]
                else:
                    I[i][j][id] = 1

        for i in range(num_events):
            for j in range(num_events):
                for k in range(num_events):
                    if i != j != k:
                        model += I[i][j][a] + I[j][k][a] - I[i][k][a] <= 1
                        model += I[i][j][a] + I[j][k][d] - I[i][k][a] - I[i][k][d] <= 1
                        model += I[i][j][a] + I[j][k][di] - I[i][k][a] <= 1
                        model += I[i][j][a] + I[j][k][id] - I[i][k][a] <= 1

                        model += I[i][j][b] + I[j][k][b] - I[i][k][b] <= 1
                        model += I[i][j][b] + I[j][k][d] - I[i][k][b] - I[i][k][d] <= 1
                        model += I[i][j][b] + I[j][k][di] - I[i][k][b] <= 1
                        model += I[i][j][b] + I[j][k][id] - I[i][k][b] <= 1

                        model += I[i][j][d] + I[j][k][a] - I[i][k][a] <= 1
                        model += I[i][j][d] + I[j][k][b] - I[i][k][b] <= 1
                        model += I[i][j][d] + I[j][k][d] - I[i][k][d] <= 1
                        model += I[i][j][d] + I[j][k][id] - I[i][k][d] <= 1

                        model += I[i][j][di] + I[j][k][a] - I[i][k][a] - I[i][k][di] <= 1
                        model += I[i][j][di] + I[j][k][b] - I[i][k][b] - I[i][k][di] <= 1
                        model += I[i][j][di] + I[j][k][di] - I[i][k][di] <= 1
                        model += I[i][j][di] + I[j][k][id] - I[i][k][di] <= 1

                        model += I[i][j][id] + I[j][k][a] - I[i][k][a] <= 1
                        model += I[i][j][id] + I[j][k][b] - I[i][k][b] <= 1
                        model += I[i][j][id] + I[j][k][d] - I[i][k][d] <= 1
                        model += I[i][j][id] + I[j][k][di] - I[i][k][di] <= 1
                        model += I[i][j][id] + I[j][k][id] - I[i][k][id] <= 1

        # Solve problem
        # print(model)
        # model.writeLP("OptProblem.lp")
        print("SOLVING OPTIMIZATION PROBLEM")
        model.solve(pulp.solvers.CPLEX_PY())


        print("SOLVED OPTIMIZATION PROBLEM")

        print("Status:", pulp.LpStatus[model.status])
        rel_pred = []
        for i in range(num_events):
            rel_pred.append([])
            for j in range(num_events):
                rel_pred[i].append([])
                for k in range(self.num_interval_relations):
                    rel_pred[i][j].append(pulp.value(I[i][j][k]))

        rel_pred = np.array(rel_pred)

        # self.generate_final_prediction_file(events,rel_pred)
        print("predicted relations dimentions :", rel_pred.shape)
        print("#$#$" * 20)
        return events, rel_pred

    def get_end_point_score(self,probability_score):

        a = 0
        b = 1
        d = 2
        di = 3
        id = 4
        v = 5

        # 'AFTER': [0, 0, 0, 0],
        # 'BEFORE': [1, 1, 1, 1],
        # 'DURING': [0, 1, 1, 0],
        # 'IS_INCLUDED': [0, 1, 1, 0],
        # 'INCLUDES': [1, 0, 1, 0],
        # 'DURING_INV': [1, 0, 1, 0],
        # 'IDENTITY': [2, 2, 1, 0],
        # 'SIMULTANEOUS': [2, 2, 1, 0],

        new_prob_score = []

        num_base_point_rel = 3

        ev_pair = probability_score.shape[0]

        print(probability_score.shape)

        for ind in range(ev_pair):
            # print("----"*10)

            # print(probability_score[ind])

            end_points_score_1 = [0]*num_base_point_rel
            end_points_score_2 = [0]*num_base_point_rel
            end_points_score_3 = [0]*num_base_point_rel
            end_points_score_4 = [0]*num_base_point_rel

            end_points_score_1[0] = probability_score[ind][a] + probability_score[ind][d]
            end_points_score_1[1] = probability_score[ind][b] + probability_score[ind][di]
            end_points_score_1[2] = probability_score[ind][id]

            end_points_score_2[0] = probability_score[ind][a] + probability_score[ind][di]
            end_points_score_2[1] = probability_score[ind][b] + probability_score[ind][d]
            end_points_score_2[2] = probability_score[ind][id]

            end_points_score_3[0] = probability_score[ind][a]
            end_points_score_3[1] = probability_score[ind][b] + probability_score[ind][d] + probability_score[ind][di] +probability_score[ind][id]

            end_points_score_4[0] = probability_score[ind][a]+ probability_score[ind][d] + probability_score[ind][di] + probability_score[ind][id]
            end_points_score_4[1] = probability_score[ind][b]

            end_point_scores_for_this_pair = [end_points_score_1,end_points_score_2,end_points_score_3,end_points_score_4]
            # print(end_point_scores_for_this_pair)
            new_prob_score.append(end_point_scores_for_this_pair)

        new_prob_score = np.array(new_prob_score)
        print(new_prob_score.shape)
        return new_prob_score



    def optimize_exact_point_relations(self, event_pairs, probability_score):

        # print("length of list",len(probability_score))
        # for i in range(len(probability_score)):
        #     print("Point relation data " + str(i) + " : shape : " + str(probability_score[i].shape))

        '''end_point e^- will be present at even location in the list and e^+ at odd.
           e.g. e^-_1 will be at 0 th location and e^+_1 at 1 location in list.

        '''

        if len(probability_score.shape)==2:
            probability_score = self.get_end_point_score(probability_score)

        start_start = 0
        end_end = 1
        end_start = 2
        start_end = 3

        a = 0
        b = 1
        id = 2
        leq = 3
        geq = 4

        def _determine_which_end_point_pairs(i, j):

            is_end = lambda x: x % 2  # if its odd number then its end of interval

            if is_end(i):
                if is_end(j):
                    return end_end

                else:
                    return end_start

            else:
                if is_end(j):
                    return start_end
                else:
                    return start_start

        def __get_proba_score(i, j):

            ev1_ind = i / 2
            ev2_ind = j / 2

            e1 = events[ev1_ind]
            e2 = events[ev2_ind]
            proba_score = [0] * self.num_base_point_relations
            if i == j:
                proba_score[id] = 1

            elif ev1_ind == ev2_ind:
                if i - j == 1:
                    proba_score[a] = 1
                else:
                    proba_score[b] = 1

            else:
                pair = e1 + feat_separator + e2
                if ev_pairs_rel_dict.get(pair, None) is not None:
                    # print("Complete score list for the pair is", ev_pairs_rel_dict[pair])
                    proba_score = ev_pairs_rel_dict[pair][_determine_which_end_point_pairs(i, j)]

            # print("For indices i,j : {0} and {1}, event pairs are {2} {3} and score is {4}".format(i,j,e1,e2,proba_score))

            # print("----"*10)

            return proba_score

        def __get_reverse_score(prob_score):
            rev_prob_score = []
            for end_point_score in prob_score:
                rev_score = []
                rev_score.append(end_point_score[b])
                rev_score.append(end_point_score[a])
                rev_score.append(end_point_score[id])
                rev_prob_score.append(rev_score)
            return rev_prob_score

        events = []
        ev_pairs_rel_dict = {}
        for ind, line in enumerate(event_pairs):
            # print("-----" * 10)
            line = line.strip().split(feat_separator)
            e1 = line[0]
            e2 = line[1]
            p_score = probability_score[ind].tolist()
            rev_p_score = __get_reverse_score(p_score)

            pair = e1+feat_separator+e2
            rev_pair = e2+feat_separator+e1

            ev_pairs_rel_dict[pair] = p_score
            ev_pairs_rel_dict[rev_pair] = rev_p_score

            # print("For event pair {0} and {1}, probability score is {2}".format(e1,e2,p_score))
            # print("For event reverse pair {0} and {1}, probability score is {2}".format(e2,e1,rev_p_score))

            events.append(e1)
            events.append(e2)

        events = sorted(set(events))
        num_events = len(events)
        num_end_points = 2 * num_events
        print("Number of events", num_events)

        # probability score
        p = []
        for i in range(num_end_points):
            p.append([])
            for j in range(num_end_points):
                p[i].append([])
                p_score = __get_proba_score(i, j)
                # print(p_score)
                p[i][j] = p_score

        print("Total number of variables : ",num_end_points*num_end_points*self.num_base_point_relations)

        # defining indicator variable which can take value 0 or 1
        I = pulp.LpVariable.dicts('I', (range(num_end_points), range(num_end_points), range(self.num_base_point_relations)),
                                  lowBound=0, upBound=1,
                                  cat=pulp.LpInteger)

        model = pulp.LpProblem("Maximizing score", pulp.LpMaximize)

        # Objective Function
        model += (
            pulp.lpSum(
                [p[i][j][k] * I[i][j][k] for k in range(self.num_base_point_relations) for j in range(num_end_points) for i in range(num_end_points)])
        )

        # constraints

        for i in range(num_end_points):
            for j in range(num_end_points):
                model += (pulp.lpSum([I[i][j][k] for k in range(self.num_base_point_relations)]) == 1)
                if i != j:
                    model += I[i][j][a] == I[j][i][b]
                    # model += I[i][j][leq] == I[j][i][geq]
                    model += I[i][j][id] == I[j][i][id]
                else:
                    I[i][j][id] = 1

        for i in range(num_end_points):
            for j in range(num_end_points):
                for k in range(num_end_points):
                    if i != j != k:
                        model += I[i][j][a] + I[j][k][a] - I[i][k][a] <= 1
                        # model += I[i][j][a] + I[j][k][geq] - I[i][k][a] <= 1
                        model += I[i][j][a] + I[j][k][id] - I[i][k][a] <= 1

                        model += I[i][j][b] + I[j][k][b] - I[i][k][b] <= 1
                        # model += I[i][j][b] + I[j][k][leq] - I[i][k][b] <= 1
                        model += I[i][j][b] + I[j][k][id] - I[i][k][b] <= 1

                        # model += I[i][j][leq] + I[j][k][leq] - I[i][k][leq] <= 1
                        # model += I[i][j][leq] + I[j][k][b] - I[i][k][b] <= 1
                        # model += I[i][j][leq] + I[j][k][id] - I[i][k][leq] <= 1
                        #
                        # model += I[i][j][geq] + I[j][k][a] - I[i][k][a] <= 1
                        # model += I[i][j][geq] + I[j][k][geq] - I[i][k][geq] <= 1
                        # model += I[i][j][geq] + I[j][k][id] - I[i][k][geq] <= 1
                        #
                        model += I[i][j][id] + I[j][k][a] - I[i][k][a] <= 1
                        model += I[i][j][id] + I[j][k][b] - I[i][k][b] <= 1
                        # model += I[i][j][id] + I[j][k][leq] - I[i][k][leq] <= 1
                        # model += I[i][j][id] + I[j][k][geq] - I[i][k][geq] <= 1
                        model += I[i][j][id] + I[j][k][id] - I[i][k][id] <= 1

                        # model += I[i][j][leq] + I[i][j][a] <= 1
                        # model += I[i][j][geq] + I[i][j][b] <= 1
                        # model += I[i][j][leq] + I[i][j][geq] - I[i][j][id] <= 1
                        #
                        # model += I[i][j][leq] + I[i][j][geq] - I[i][j][id] <= 1

        # Solve problem
        # print(model)
        # model.writeLP("OptProblem.lp")
        print("SOLVING OPTIMIZATION PROBLEM")
        # model.solve(pulp.solvers.PULP_CBC_CMD())
        model.solve(pulp.solvers.CPLEX_PY())

        print("SOLVED OPTIMIZATION PROBLEM")

        print("Status:", pulp.LpStatus[model.status])
        point_rel_pred = []
        for i in range(num_end_points):
            point_rel_pred.append([])
            for j in range(num_end_points):
                point_rel_pred[i].append([])
                for k in range(self.num_base_point_relations):
                    point_rel_pred[i][j].append(pulp.value(I[i][j][k]))

        point_rel_pred = np.array(point_rel_pred)

        print("predicted relations dimentions :", point_rel_pred.shape)

        rel_pred = []
        for i in range(num_events):
            rel_pred.append([])
            for j in range(num_events):
                rel_pred[i].append([])
                if i != j:
                    end_point_rel_list = [0] * 4
                    ev1_start_point = i * 2
                    ev1_end_point = i * 2 + 1
                    ev2_start_point = j * 2
                    ev2_end_point = j * 2 + 1

                    end_point_rel_list[start_start] = point_rel_pred[ev1_start_point][ev2_start_point]
                    end_point_rel_list[start_end] = point_rel_pred[ev1_start_point][ev2_end_point]
                    end_point_rel_list[end_start] = point_rel_pred[ev1_end_point][ev2_start_point]
                    end_point_rel_list[end_end] = point_rel_pred[ev1_end_point][ev2_end_point]

                    rel_pred[i][j] = end_point_rel_list

        rel_pred = np.array(rel_pred)

        print("predicted relations dimentions :", rel_pred.shape)
        print(rel_pred[0][1])
        return events, rel_pred

    def convert_point_to_interval_relations(self, point_relations):

        point_int_conv_dict = {"1231": "a"}
        a = 0
        b = 1
        id = 2
        leq = 3
        geq = 4

        vague = 5
        self._end_point_relation_dict = {(0, 0, 0, 0):0,(1, 1, 1, 1):1,(0, 1, 1, 0):2,(1, 0, 1, 0):3,(2, 2, 1, 0):4}

        def _get_single_point_rel(rel):
            rel = np.array(rel)
            ind = np.where(rel == 1)[0]
            if len(ind) > 1:
                if geq in ind:
                    return b
                elif leq in ind:
                    return a
                else:
                    return vague
            elif len(ind) == 1:
                return ind[0]
            else:
                return vague

        # start_start_rel, start_end_rel, end_start_rel, end_end_rel
        def _get_interval_relation(end_point_rel_prob_list):
            end_point_rel_list = tuple([_get_single_point_rel(rel) for rel in end_point_rel_prob_list])
            interval_rel = vague
            if self._end_point_relation_dict.get(end_point_rel_list, None) is not None:
                interval_rel = self._end_point_relation_dict.get(end_point_rel_list)

            return interval_rel

        start_start = 0
        end_end = 1
        end_start = 2
        start_end = 3

        point_inte_conv_list = [start_start, end_end, end_start, start_end]

        point_relations = np.array(point_relations)
        num_events = point_relations.shape[0]
        print(num_events)
        rel_pred = []

        for i in range(num_events):
            rel_pred.append([])
            for j in range(num_events):
                rel_pred[i].append([])
                if i != j:
                    end_point_rel_list = []
                    for pp in point_inte_conv_list:
                        # end_point_rel_list.append(end_point_rel_list[pp])
                        end_point_rel_list.append(point_relations[i,j][pp])
                    print "end point relation ",end_point_rel_list
                    interval_rel = _get_interval_relation(end_point_rel_list)
                    print "interval relation ",interval_rel
                    rel_pred[i][j] = interval_rel
        rel_pred = np.array(rel_pred)
        print("predicted relations dimentions :", rel_pred.shape)
        return rel_pred

    def optimize_lagrangian_relaxed_point_relations(self,event_pairs, probability_score):
        # print("length of list",len(probability_score))
        # for i in range(len(probability_score)):
        #     print("Point relation data " + str(i) + " : shape : " + str(probability_score[i].shape))

        '''end_point e^- will be present at even location in the list and e^+ at odd.
           e.g. e^-_1 will be at 0 th location and e^+_1 at 1 location in list.

        '''

        is_solve_with_cplex = False

        if len(probability_score.shape)==2:
            probability_score = self.get_end_point_score(probability_score)

        start_start = 0
        end_end = 1
        end_start = 2
        start_end = 3

        a = 0
        b = 1
        id = 2
        leq = 3
        geq = 4




        def _multiplier_update(lamda, step_size, sub_grad):
            lamda = lamda - step_size * sub_grad
            return lamda if lamda > 0 else 0

        def _determine_which_end_point_pairs(i, j):

            is_end = lambda x: x % 2  # if its odd number then its end of interval

            if is_end(i):
                if is_end(j):
                    return end_end

                else:
                    return end_start

            else:
                if is_end(j):
                    return start_end
                else:
                    return start_start

        def __get_proba_score(i, j):

            ev1_ind = i / 2
            ev2_ind = j / 2

            e1 = events[ev1_ind]
            e2 = events[ev2_ind]
            proba_score = [0] * self.num_base_point_relations
            if i == j:
                proba_score[id] = 1

            elif ev1_ind == ev2_ind:
                if i - j == 1:
                    proba_score[a] = 1
                else:
                    proba_score[b] = 1

            else:
                pair = e1 + feat_separator + e2
                if ev_pairs_rel_dict.get(pair, None) is not None:
                    # print("Complete score list for the pair is", ev_pairs_rel_dict[pair])
                    proba_score = ev_pairs_rel_dict[pair][_determine_which_end_point_pairs(i, j)]

            # print("For indices i,j : {0} and {1}, event pairs are {2} {3} and score is {4}".format(i,j,e1,e2,proba_score))

            # print("----"*10)

            return proba_score

        def __get_reverse_score(prob_score):
            rev_prob_score = []
            for end_point_score in prob_score:
                rev_score = []
                rev_score.append(end_point_score[b])
                rev_score.append(end_point_score[a])
                rev_score.append(end_point_score[id])
                rev_prob_score.append(rev_score)
            return rev_prob_score


        def _get_pulp_value(i1_):
            try:
                return 0 if pulp.value(i1_) is None else pulp.value(i1_)
            except:
                return i1_


        def _calculate_sub_grad(i1,i2,i3):
            return 1- _get_pulp_value(i1) - _get_pulp_value(i2) + _get_pulp_value(i3)

        def _compare_arrays(a, b):
            is_same = True
            if a.shape[0] == 4:
                for i in range(4):
                    if not (a[i] == b[i]).all():
                        return False
            else:
                for i in range(5):
                    if a[i]!=b[i]:
                        return False
            return True

        def calc_curr_step_size(I_indicator_func, lamda_multipliers, best_known_bound):
            sub_grad = np.zeros((num_constraints_relaxed, num_end_points, num_end_points, num_end_points))
            alpha = 1.5
            for i in range(num_end_points):
                for j in range(num_end_points):
                    for k in range(num_end_points):
                        sub_grad[0][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][b],
                                                                   I_indicator_func[j][k][b],
                                                                   I_indicator_func[i][k][b])
                        sub_grad[1][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][b],
                                                                   I_indicator_func[j][k][leq],
                                                                   I_indicator_func[i][k][b])
                        sub_grad[2][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][b],
                                                                   I_indicator_func[j][k][id],
                                                                   I_indicator_func[i][k][b])

                        sub_grad[3][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][leq],
                                                                   I_indicator_func[j][k][b],
                                                                   I_indicator_func[i][k][b])
                        sub_grad[4][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][leq],
                                                                   I_indicator_func[j][k][leq],
                                                                   I_indicator_func[i][k][leq])
                        sub_grad[5][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][leq],
                                                                   I_indicator_func[j][k][id],
                                                                   I_indicator_func[i][k][leq])

                        sub_grad[6][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][a],
                                                                   I_indicator_func[j][k][a],
                                                                   I_indicator_func[i][k][a])
                        sub_grad[7][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][a],
                                                                   I_indicator_func[j][k][geq],
                                                                   I_indicator_func[i][k][a])
                        sub_grad[8][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][a],
                                                                   I_indicator_func[j][k][id],
                                                                   I_indicator_func[i][k][a])

                        sub_grad[9][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][geq],
                                                                   I_indicator_func[j][k][a],
                                                                   I_indicator_func[i][k][a])
                        sub_grad[10][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][geq],
                                                                    I_indicator_func[j][k][geq],
                                                                    I_indicator_func[i][k][geq])
                        sub_grad[11][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][geq],
                                                                    I_indicator_func[j][k][id],
                                                                    I_indicator_func[i][k][geq])

                        sub_grad[12][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][id],
                                                                    I_indicator_func[j][k][b],
                                                                    I_indicator_func[i][k][b])
                        sub_grad[13][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][id],
                                                                    I_indicator_func[j][k][a],
                                                                    I_indicator_func[i][k][a])
                        sub_grad[14][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][id],
                                                                    I_indicator_func[j][k][id],
                                                                    I_indicator_func[i][k][id])
                        sub_grad[15][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][id],
                                                                    I_indicator_func[j][k][leq],
                                                                    I_indicator_func[i][k][leq])
                        sub_grad[16][i][j][k] = _calculate_sub_grad(I_indicator_func[i][j][id],
                                                                    I_indicator_func[j][k][geq],
                                                                    I_indicator_func[i][k][geq])

            curr_obj_val = 0

            for i in range(num_end_points):
                for j in range(num_end_points):
                    if i != j:
                        for r in range(self.num_base_point_relations):
                            curr_obj_val = curr_obj_val + p[i][j][r] * I_indicator_func[i][j][r]
                        for k in range(num_end_points):
                            curr_obj_val = curr_obj_val + lamda_multipliers[0][i][j][k] * (
                                    1 - I_indicator_func[i][j][b] - I_indicator_func[j][k][b] + I_indicator_func[i][k][
                                b])
                            + lamda_multipliers[1][i][j][k] * (
                                    1 - I_indicator_func[i][j][b] - I_indicator_func[j][k][leq] +
                                    I_indicator_func[i][k][b])
                            + lamda_multipliers[2][i][j][k] * (
                                    1 - I_indicator_func[i][j][b] - I_indicator_func[j][k][id] + I_indicator_func[i][k][
                                b])
                            + lamda_multipliers[3][i][j][k] * (
                                    1 - I_indicator_func[i][j][leq] - I_indicator_func[j][k][b] +
                                    I_indicator_func[i][k][b])
                            + lamda_multipliers[4][i][j][k] * (
                                    1 - I_indicator_func[i][j][leq] - I_indicator_func[j][k][leq] +
                                    I_indicator_func[i][k][leq])
                            + lamda_multipliers[5][i][j][k] * (
                                    1 - I_indicator_func[i][j][leq] - I_indicator_func[j][k][id] +
                                    I_indicator_func[i][k][leq])
                            + lamda_multipliers[6][i][j][k] * (
                                    1 - I_indicator_func[i][j][a] - I_indicator_func[j][k][a] + I_indicator_func[i][k][
                                a])
                            + lamda_multipliers[7][i][j][k] * (
                                    1 - I_indicator_func[i][j][a] - I_indicator_func[j][k][geq] +
                                    I_indicator_func[i][k][a])
                            + lamda_multipliers[8][i][j][k] * (
                                    1 - I_indicator_func[i][j][a] - I_indicator_func[j][k][id] + I_indicator_func[i][k][
                                a])
                            + lamda_multipliers[9][i][j][k] * (
                                    1 - I_indicator_func[i][j][geq] - I_indicator_func[j][k][a] +
                                    I_indicator_func[i][k][a])
                            + lamda_multipliers[10][i][j][k] * (
                                    1 - I_indicator_func[i][j][geq] - I_indicator_func[j][k][geq] +
                                    I_indicator_func[i][k][geq])
                            + lamda_multipliers[11][i][j][k] * (
                                    1 - I_indicator_func[i][j][geq] - I_indicator_func[j][k][id] +
                                    I_indicator_func[i][k][geq])
                            + lamda_multipliers[12][i][j][k] * (
                                    1 - I_indicator_func[i][j][id] - I_indicator_func[j][k][b] + I_indicator_func[i][k][
                                b])
                            + lamda_multipliers[13][i][j][k] * (
                                    1 - I_indicator_func[i][j][id] - I_indicator_func[j][k][a] + I_indicator_func[i][k][
                                a])
                            + lamda_multipliers[14][i][j][k] * (
                                    1 - I_indicator_func[i][j][id] - I_indicator_func[j][k][id] +
                                    I_indicator_func[i][k][id])
                            + lamda_multipliers[15][i][j][k] * (
                                    1 - I_indicator_func[i][j][id] - I_indicator_func[j][k][leq] +
                                    I_indicator_func[i][k][leq])
                            + lamda_multipliers[16][i][j][k] * (
                                    1 - I_indicator_func[i][j][id] - I_indicator_func[j][k][geq] +
                                    I_indicator_func[i][k][geq])

            sub_grad_l2norm_sqaure = (LA.norm(sub_grad)) ** 2

            if best_known_bound is None:
                # step_size = random.uniform(0, 1)
                step_size = 0.1
                best_known_bound = curr_obj_val
            else:
                step_size = alpha * (best_known_bound - curr_obj_val) / sub_grad_l2norm_sqaure

            for i in range(num_end_points):
                for j in range(num_end_points):
                    for k in range(num_end_points):
                        for l in range(num_constraints_relaxed):
                            lamda_multipliers[l][i][j][k] = _multiplier_update(lamda_multipliers[l][i][j][k], step_size,
                                                                               sub_grad[l][i][j][k])

            print "current objective value",curr_obj_val
            print "sub gradient l2norm square",sub_grad_l2norm_sqaure

            print "best bound ",best_known_bound
            print "step size",step_size

            print "----"*10

            if curr_obj_val < best_known_bound:
                best_known_bound = curr_obj_val

            return best_known_bound, lamda_multipliers


        events = []
        ev_pairs_rel_dict = {}
        for ind, line in enumerate(event_pairs):
            # print("-----" * 10)
            line = line.strip().split(feat_separator)
            e1 = line[0]
            e2 = line[1]
            p_score = probability_score[ind].tolist()
            rev_p_score = __get_reverse_score(p_score)

            pair = e1+feat_separator+e2
            rev_pair = e2+feat_separator+e1

            ev_pairs_rel_dict[pair] = p_score
            ev_pairs_rel_dict[rev_pair] = rev_p_score

            # print("For event pair {0} and {1}, probability score is {2}".format(e1,e2,p_score))
            # print("For event reverse pair {0} and {1}, probability score is {2}".format(e2,e1,rev_p_score))

            events.append(e1)
            events.append(e2)

        events = sorted(set(events))
        events = events[0:2]
        num_events = len(events)
        # num_events = 5
        num_end_points = 2 * num_events
        print("Number of events", num_events)

        # probability score
        p = []
        for i in range(num_end_points):
            p.append([])
            for j in range(num_end_points):
                p[i].append([])
                p_score = __get_proba_score(i, j)
                # print(p_score)
                p[i][j] = p_score

        print("Total number of variables : ",num_end_points*num_end_points*self.num_base_point_relations)

        # defining indicator variable which can take value 0 or 1
        # if is_solve_with_cplex:
        #     I = pulp.LpVariable.dicts('I', (range(num_end_points), range(num_end_points), range(self.num_point_relations)),
        #                           lowBound=0, upBound=1,
        #                           cat=pulp.LpInteger)
        #
        # else:
        #     I = np.zeros((num_end_points, num_end_points, self.num_point_relations))

        # I = pulp.LpVariable.dicts('I', (range(num_end_points), range(num_end_points), range(self.num_point_relations)),
        #                           lowBound=0, upBound=1,
        #                           cat=pulp.LpInteger)

        num_constraints_relaxed = 17
        lamda = np.random.random((num_constraints_relaxed, num_end_points, num_end_points,num_end_points)) *0 # initialize as 0
        sub_grad = np.random.randint(1, size=(num_constraints_relaxed, num_end_points, num_end_points,num_end_points))*0

        # I_analytical = np.zeros((num_end_points, num_end_points, self.num_point_relations))
        #
        # I = np.zeros((num_end_points, num_end_points, self.num_point_relations))


        I_analytical = np.ones((num_end_points, num_end_points, self.num_point_relations))

        I = np.ones((num_end_points, num_end_points, self.num_point_relations))

        lamda_analytical = np.random.random((num_constraints_relaxed, num_end_points, num_end_points, num_end_points)) * 0  # initialize as 0
        sub_grad_analytical = np.random.randint(1, size=(
        num_constraints_relaxed, num_end_points, num_end_points, num_end_points)) * 0

        for i in range(num_end_points):
            for j in range(num_end_points):
                if i ==j :
                    I[i][j][:] = 0
                    I_analytical[i][j][:] = 0

                    I[i][j][id] =1
                    I_analytical[i][j][id] = 1

                    I_analytical[i][j][leq] = 1
                    I_analytical[i][j][geq] = 1

                    I[i][j][leq] = 1
                    I[i][j][geq] = 1


        I_direct = pulp.LpVariable.dicts('I', (
        range(num_end_points), range(num_end_points), range(self.num_point_relations)),
                                         lowBound=0, upBound=1,
                                         cat=pulp.LpInteger)
        print (I_direct)
        print (p)
        lamda_direct = np.random.random((num_constraints_relaxed, num_end_points, num_end_points, num_end_points)) * 0

        MAX_ITER = 10
        total_opt_problems = num_end_points * num_end_points - num_end_points
        # best_known_bound =sys.maxint
        # best_known_bound_analytical = sys.maxint

        best_known_bound =None
        best_known_bound_analytical = None
        best_known_bound_direct = None
        inv_dict = {a: b, b: a, id: id, geq: leq, leq: geq}
        for ite in range(MAX_ITER):
            count = 0
            mis_match_count =0
            for i in range(num_end_points):
                for j in range(i+1,num_end_points):
                    if i != j:

                        lamda_sum_i_j_k = [0] * self.num_point_relations
                        lamda_sum_k_i_j = [0] * self.num_point_relations
                        lamda_sum_i_k_j = [0] * self.num_point_relations

                        for k in range(num_end_points):
                            lamda_sum_i_j_k[a] = lamda_sum_i_j_k[a] + lamda[6][i][j][k] + lamda[7][i][j][k] + \
                                                 lamda[8][i][j][k]
                            lamda_sum_i_j_k[b] = lamda_sum_i_j_k[b] + lamda[0][i][j][k] + lamda[1][i][j][k] + \
                                                 lamda[2][i][j][k]
                            lamda_sum_i_j_k[id] = lamda_sum_i_j_k[id] + lamda[12][i][j][k] + lamda[13][i][j][k] + \
                                                  lamda[14][i][j][k] + lamda[15][i][j][k] + lamda[16][i][j][k]
                            lamda_sum_i_j_k[leq] = lamda_sum_i_j_k[leq] + lamda[3][i][j][k] + lamda[4][i][j][k] + \
                                                   lamda[5][i][j][k]
                            lamda_sum_i_j_k[geq] = lamda_sum_i_j_k[geq] + lamda[9][i][j][k] + lamda[10][i][j][k] + \
                                                   lamda[11][i][j][k]

                            lamda_sum_k_i_j[a] = lamda_sum_k_i_j[a] + lamda[6][k][i][j] + lamda[9][k][i][j] + \
                                                 lamda[13][k][i][j]
                            lamda_sum_k_i_j[b] = lamda_sum_k_i_j[b] + lamda[0][k][i][j] + lamda[3][k][i][j] + \
                                                 lamda[12][k][i][j]
                            lamda_sum_k_i_j[id] = lamda_sum_k_i_j[id] + lamda[2][k][i][j] + lamda[5][k][i][j] + \
                                                  lamda[8][k][i][j] + \
                                                  lamda[11][k][i][j] + lamda[14][k][i][j]
                            lamda_sum_k_i_j[leq] = lamda_sum_k_i_j[leq] + lamda[1][k][i][j] + lamda[4][k][i][j] + \
                                                   lamda[15][k][i][j]
                            lamda_sum_k_i_j[geq] = lamda_sum_k_i_j[geq] + lamda[7][k][i][j] + lamda[10][k][i][j] + \
                                                   lamda[16][k][i][j]

                            lamda_sum_i_k_j[a] = lamda_sum_i_k_j[a] + lamda[6][i][k][j] + lamda[7][i][k][j] + \
                                                 lamda[8][i][k][j] + lamda[9][i][k][j] + lamda[13][i][k][j]

                            lamda_sum_i_k_j[b] = lamda_sum_i_k_j[b] + lamda[0][i][k][j] + lamda[1][i][k][j] + \
                                                 lamda[2][i][k][j] + lamda[3][i][k][j] + lamda[12][i][k][j]

                            lamda_sum_i_k_j[id] = lamda_sum_i_k_j[id] + lamda[14][i][k][j]
                            lamda_sum_i_k_j[leq] = lamda_sum_i_k_j[leq] + lamda[4][i][k][j] + lamda[5][i][k][j] + \
                                                   lamda[15][i][k][j]
                            lamda_sum_i_k_j[geq] = lamda_sum_i_k_j[geq] + lamda[10][i][k][j] + lamda[11][i][k][j] + \
                                                   lamda[16][i][k][j]

                        support_value = [0] * self.num_point_relations
                        for r in range(self.num_point_relations):
                            if r > 2:
                                cls_score = 0
                            else:
                                cls_score = p[i][j][r]
                            support_value[r] = cls_score + lamda_sum_i_k_j[r] - lamda_sum_i_j_k[r] - lamda_sum_k_i_j[r]

                        s_ij = pulp.LpVariable.dicts('s_ij',(range(self.num_point_relations)),lowBound=0, upBound=1,cat=pulp.LpInteger)

                        model = pulp.LpProblem("Maximizing score", pulp.LpMaximize)

                        # Objective Function
                        model += (pulp.lpSum([support_value[r] * s_ij[r] for r in range(self.num_point_relations)]))


                        # constraints

                        #at most one base relation
                        model += (pulp.lpSum([s_ij[r] for r in range(self.num_base_point_relations)]) <= 1)

                        model += s_ij[leq] + s_ij[a] <= 1
                        model += s_ij[geq] + s_ij[b] <= 1
                        model += s_ij[leq] + s_ij[geq] - s_ij[id] <= 1

                        model += s_ij[b] + s_ij[id] <= s_ij[leq]
                        model += s_ij[a] + s_ij[id] <= s_ij[geq]

                        # Solve problem
                        print(model)
                        # model.writeLP("OptProblem.lp")
                        print("SOLVING OPTIMIZATION PROBLEM")
                        # model.solve(pulp.solvers.PULP_CBC_CMD())
                        model.solve(pulp.solvers.CPLEX_PY())

                        print("SOLVED OPTIMIZATION PROBLEM WITH CPLEX")

                        print("Status:", pulp.LpStatus[model.status])

                        for r in range(self.num_point_relations):
                            I[i][j][r] = _get_pulp_value(s_ij[r])
                            I[j][i][r] = _get_pulp_value(s_ij[inv_dict[r]])

                        print("SOLVING BY INSPECTION")

                        max_ind = -1

                        for r in range(self.num_point_relations):
                            I_analytical[i][j][r] = 0
                            I_analytical[j][i][r] = 0

                        id_score = support_value[id]
                        b_score = support_value[b] + support_value[leq]
                        a_score = support_value[a] + support_value[geq]

                        leq_score =  support_value[leq]
                        geq_score = support_value[geq]

                        scores_list = [a_score, b_score, id_score,leq_score,geq_score]

                        if max(scores_list)>0:
                            max_ind = scores_list.index(max(scores_list))

                        if max_ind == a:
                            I_analytical[i][j][a] = 1
                            I_analytical[i][j][geq] = 1

                            I_analytical[j][i][b] = 1
                            I_analytical[j][i][leq] = 1

                        elif max_ind == b:
                            I_analytical[i][j][b] = 1
                            I_analytical[i][j][leq] = 1

                            I_analytical[j][i][a] = 1
                            I_analytical[j][i][geq] = 1

                        elif max_ind == id:
                            I_analytical[i][j][id] = 1
                            I_analytical[j][i][id] = 1

                        elif max_ind == leq :
                            I_analytical[i][j][leq] = 1
                            I_analytical[j][i][geq] = 1

                        elif max_ind == geq:
                            I_analytical[i][j][geq] = 1
                            I_analytical[j][i][leq] = 1

                        print("SOLVED BY INSPECTION")


                        print "currently solving for pair {}, {}...comparison with other values also...".format(i,j)
                        print "probability score is {}".format(p[i][j])
                        print "inspection scores are  {}".format(scores_list)
                        print "solution by cplex is {} ".format(I[i][j])
                        print "solution by analysis is {} ".format(I_analytical[i][j])
                        print "Is solution same from both methods : {} ".format(_compare_arrays(I[i][j],I_analytical[i][j]))


                        count+=1
                        print "Solved {} ; remaining {} to solve".format(count,total_opt_problems-count)
                        print ("$$$$"*20)

                        # if mis_match_count >9:
                        #     sys.exit("Exitting as 10 mismatch")

            model_direct = pulp.LpProblem("Maximizing score", pulp.LpMaximize)

            # Objective Function

            model_direct += (pulp.lpSum([p[i][j][r] * I_direct[i][j][r] for r in range(self.num_base_point_relations) for i in
                                  range(num_end_points) for j in range(i+1,num_end_points)] + [lamda_direct[0][i][j][k] * (
                                              1 - I_direct[i][j][b] - I_direct[j][k][b] + I_direct[i][k][b])
                                  + lamda_direct[1][i][j][k] * (
                                              1 - I_direct[i][j][b] - I_direct[j][k][leq] + I_direct[i][k][b])
                                  + lamda_direct[2][i][j][k] * (
                                              1 - I_direct[i][j][b] - I_direct[j][k][id] + I_direct[i][k][b])
                                  + lamda_direct[3][i][j][k] * (
                                              1 - I_direct[i][j][leq] - I_direct[j][k][b] + I_direct[i][k][b])
                                  + lamda_direct[4][i][j][k] * (
                                              1 - I_direct[i][j][leq] - I_direct[j][k][leq] + I_direct[i][k][leq])
                                  + lamda_direct[5][i][j][k] * (
                                              1 - I_direct[i][j][leq] - I_direct[j][k][id] + I_direct[i][k][leq])
                                  + lamda_direct[6][i][j][k] * (
                                              1 - I_direct[i][j][a] - I_direct[j][k][a] + I_direct[i][k][a])
                                  + lamda_direct[7][i][j][k] * (
                                              1 - I_direct[i][j][a] - I_direct[j][k][geq] + I_direct[i][k][a])
                                  + lamda_direct[8][i][j][k] * (
                                              1 - I_direct[i][j][a] - I_direct[j][k][id] + I_direct[i][k][a])
                                  + lamda_direct[9][i][j][k] * (
                                              1 - I_direct[i][j][geq] - I_direct[j][k][a] + I_direct[i][k][a])
                                  + lamda_direct[10][i][j][k] * (
                                              1 - I_direct[i][j][geq] - I_direct[j][k][geq] + I_direct[i][k][geq])
                                  + lamda_direct[11][i][j][k] * (
                                              1 - I_direct[i][j][geq] - I_direct[j][k][id] + I_direct[i][k][geq])
                                  + lamda_direct[12][i][j][k] * (
                                              1 - I_direct[i][j][id] - I_direct[j][k][b] + I_direct[i][k][b])
                                  + lamda_direct[13][i][j][k] * (
                                              1 - I_direct[i][j][id] - I_direct[j][k][a] + I_direct[i][k][a])
                                  + lamda_direct[14][i][j][k] * (
                                              1 - I_direct[i][j][id] - I_direct[j][k][id] + I_direct[i][k][id])
                                  + lamda_direct[15][i][j][k] * (
                                              1 - I_direct[i][j][id] - I_direct[j][k][leq] + I_direct[i][k][leq])
                                  + lamda_direct[16][i][j][k] * (
                                              1 - I_direct[i][j][id] - I_direct[j][k][geq] + I_direct[i][k][geq])
                                  for k in range(num_end_points) for i in
                                  range(num_end_points) for j in range(i+1,num_end_points) ]))

            # constraints
            for i in range(num_end_points):
                for j in range(num_end_points):
                    # at most one base relation
                    model_direct += (pulp.lpSum([I_direct[i][j][r] for r in range(self.num_base_point_relations)]) <= 1)

                    # only compatible relations
                    model_direct += I_direct[i][j][leq] + I_direct[i][j][a] <= 1
                    model_direct += I_direct[i][j][geq] + I_direct[i][j][b] <= 1
                    model_direct += I_direct[i][j][leq] + I_direct[i][j][geq] - I_direct[i][j][id] <= 1

                    model_direct += I_direct[i][j][b] + I_direct[i][j][id] <= I_direct[i][j][leq]
                    model_direct += I_direct[i][j][a] + I_direct[i][j][id] <= I_direct[i][j][geq]

                    if i==j:
                        model_direct += I_direct[i][j][id] == 1



            print(model_direct)
            print("SOLVING OPTIMIZATION PROBLEM")

            model_direct.solve(pulp.solvers.CPLEX_PY())

            print("SOLVED OPTIMIZATION PROBLEM WITH CPLEX")


            print "comparing cplex and analytical solution after this iteration"

            I_direct_list = []
            for i in range(num_end_points):
                I_direct_list.append([])
                for j in range(num_end_points):
                    I_direct_list[i].append([])
                    point_rel_pred = []
                    for k in range(self.num_point_relations):
                        point_rel_pred.append(_get_pulp_value(I_direct[i][j][k]))
                    I_direct_list[i][j] = point_rel_pred


            for i in range(num_end_points):
                for j in range(i+1,num_end_points):
                    print "PAIR {}, {}".format(i,j)
                    print "CPLEX ",I[i][j]
                    print "ANALYTICAL ",I_analytical[i][j]
                    print "Direct ",I_direct_list[i][j]
                    print "-------"


            print "Updating multiplier values for cplex"
            best_known_bound,lamda = calc_curr_step_size(I,lamda,best_known_bound)
            print "Updating multiplier values for analytical solution"
            best_known_bound_analytical,lamda_analytical = calc_curr_step_size(I_analytical,lamda_analytical,best_known_bound_analytical)

            print "Updating multiplier values for direct solution"
            best_known_bound_direct, lamda_direct = calc_curr_step_size(I_direct_list, lamda_direct,best_known_bound_direct)
            # best_known_bound_direct, lamda_direct = calc_curr_step_size(I_analytical,lamda_direct,best_known_bound_direct)


            print("####" * 40)
            print "End of iteration",ite
            print ("$$##==**" * 20)



        print("SOLVED GLOBAL OPTIMIZATION PROBLEM")

        def _consolidate_point_rel(point_rel):
            interval_rel = []
            for i in range(num_events):
                interval_rel.append([])
                for j in range(num_events):
                    interval_rel[i].append([])
                    if True:
                        end_point_rel_list = [0] * 4
                        ev1_start_point = i * 2
                        ev1_end_point = i * 2 + 1
                        ev2_start_point = j * 2
                        ev2_end_point = j * 2 + 1

                        end_point_rel_list[start_start] = point_rel[ev1_start_point][ev2_start_point]
                        end_point_rel_list[start_end] = point_rel[ev1_start_point][ev2_end_point]
                        end_point_rel_list[end_start] = point_rel[ev1_end_point][ev2_start_point]
                        end_point_rel_list[end_end] = point_rel[ev1_end_point][ev2_end_point]

                        interval_rel[i][j] = end_point_rel_list

            return np.array(interval_rel)


        interval_rel_cplex = _consolidate_point_rel(I)
        interval_rel_analytical = _consolidate_point_rel(I_analytical)

        for i in range(num_events):
            for j in range(num_events):
                if i!=j:
                    print "PAIR {}, {}".format(i, j)

                    if _compare_arrays(interval_rel_cplex[i][j],interval_rel_analytical[i][j]):
                        print "same result from both methods"
                    else:
                        print "CPLEX : ", interval_rel_cplex[i][j]
                        print "ANALYTICAL", interval_rel_analytical[i][j]
                        print "different results"
                    print "-------"



        sys.exit("at the bottom of function; exitting")
        return events, I
        # return rel_pred

    def is_transitive_constraint_followed(self, i1, i2, i3):
        a = 0
        b = 1
        id = 2
        leq = 3
        geq = 4

        trans_constraint_followed = True


        if sum(i3)==0:
            return trans_constraint_followed



        def _is_followed(_i1, _i2, _i3):
            if _i1 + _i2 - _i3 <= 1:
                return True
            else:
                return False



        if not _is_followed(i1[a], i2[a], i3[a]): trans_constraint_followed = False
        if not _is_followed(i1[a], i2[geq], i3[a]): trans_constraint_followed = False
        if not _is_followed(i1[a], i2[id], i3[a]): trans_constraint_followed = False

        if not _is_followed(i1[b], i2[b], i3[b]): trans_constraint_followed = False
        if not _is_followed(i1[b], i2[leq], i3[b]): trans_constraint_followed = False
        if not _is_followed(i1[b], i2[id], i3[b]): trans_constraint_followed = False

        if not _is_followed(i1[leq], i2[leq], i3[leq]): trans_constraint_followed = False
        if not _is_followed(i1[leq], i2[b], i3[b]): trans_constraint_followed = False
        if not _is_followed(i1[leq], i2[id], i3[leq]): trans_constraint_followed = False

        if not _is_followed(i1[geq], i2[a], i3[a]): trans_constraint_followed = False
        if not _is_followed(i1[geq], i2[geq], i3[geq]): trans_constraint_followed = False
        if not _is_followed(i1[geq], i2[id], i3[geq]): trans_constraint_followed = False

        if not _is_followed(i1[id], i2[a], i3[a]): trans_constraint_followed = False
        if not _is_followed(i1[id], i2[b], i3[b]): trans_constraint_followed = False
        if not _is_followed(i1[id], i2[leq], i3[leq]): trans_constraint_followed = False
        if not _is_followed(i1[id], i2[geq], i3[geq]): trans_constraint_followed = False
        if not _is_followed(i1[id], i2[id], i3[id]): trans_constraint_followed = False

        return trans_constraint_followed


    def check_constraints_on_end_points(self, point_rel):

        a = 0
        b = 1
        id = 2
        leq = 3
        geq = 4

        more_than_one_base_rel_list_of_end_points = []
        trans_constr_vio_list_of_end_points = []
        inverse_constr_vio_list_of_end_points = []
        opposite_relations_list_of_end_points = []

        def _is_only_one_base_relation(rel):
            rel = rel[0:3]
            # print rel
            if sum(rel) > 1:
                return False
            else:
                return True


        def _is_inverse_relation(rel1,rel2):

            if sum(rel1) > 0 and sum(rel2) > 0:
                if not( rel1[a] == rel2[b] or rel1[a] == rel2[geq]):
                    return False

                if not(rel1[b] == rel2[a] or rel1[b] == rel2[leq]):
                    return False

                if rel1[id] != rel2[id] :
                    return False

                if not (rel1[leq] == rel2[b] or rel1[leq] == rel2[geq]):
                    return False

                if not (rel1[geq] == rel2[a] or rel1[geq] == rel2[leq]):
                    return False

                return True
            else:
                return True


        print point_rel.shape

        if point_rel.shape[0]!=point_rel.shape[1] or point_rel.shape[2]!=self.num_point_relations:
            print "Error in shape"
            sys.exit("Exitting .....")



        num_end_points = point_rel.shape[0]
        for i in range(num_end_points):
            for j in range(num_end_points):
                point_rel[i][j] = [0 if e==None else e for e in point_rel[i][j]]


        #dealing with disjunctions
        for i in range(num_end_points):
            for j in range(num_end_points):
                # if <= and >= then =
                if point_rel[i][j][leq] +point_rel[i][j][geq]>1:
                    point_rel[i][j] = point_rel[i][j] * 0
                    point_rel[i][j][id] = 1

                # completely ignoring disjunctions
                if point_rel[i][j][a] + point_rel[i][j][leq] + point_rel[i][j][geq]>1:
                    point_rel[i][j] = point_rel[i][j] * 0
                    point_rel[i][j][a] = 1

                if point_rel[i][j][b] + point_rel[i][j][leq] + point_rel[i][j][geq]>1:
                    point_rel[i][j] = point_rel[i][j] * 0
                    point_rel[i][j][b] = 1

                if i==j:
                    point_rel[i][j] = point_rel[i][j] * 0
                    point_rel[i][j][id] = 1


        for i in range(num_end_points):
            for j in range(num_end_points):
                if i!=j and sum(point_rel[i][j])>0:
                    print "PAIR {}, {}".format(i, j)
                    print "score is {} ".format(point_rel[i][j])
                    # point_rel[i][j] = [0 if e==None else e for e in point_rel[i][j]]

                    if not _is_only_one_base_relation(point_rel[i][j]):
                        # print "yes...more than one base relation"
                        more_than_one_base_rel_list_of_end_points.append((i,j))

                    if not _is_inverse_relation(point_rel[i][j],point_rel[j][i]):
                        inverse_constr_vio_list_of_end_points.append((i,j))


                    if point_rel[i][j][a] + point_rel[i][j][geq] >1 or point_rel[i][j][b] + point_rel[i][j][leq] >1:
                        # print "base and disjunctive relations are not assigned"
                        print "i,j score {}".format(point_rel[i][j])
                        opposite_relations_list_of_end_points.append((i,j))

                    print "checking transitive constraints"
                    for k in range(num_end_points):
                        if i != j != k:
                            if not self.is_transitive_constraint_followed(point_rel[i][j], point_rel[j][k],point_rel[i][k]):
                                # print "i,j score {}".format(point_rel[i][j])
                                # print "j,k score {}".format(point_rel[j][k])
                                # print "i,k score {}".format(point_rel[i][k])
                                # print "------------"
                                trans_constr_vio_list_of_end_points.append((i,j,k))

        print "more than one base relation is assigned to following pairs"
        for (i,j) in more_than_one_base_rel_list_of_end_points:
            print "{},{}".format(i,j)
            print "actual scores are {}".format(point_rel[i][j])
            print "\n"
            print "||||"*20
        print "======"*20

        print "inverse relation constraints are not followed in -"
        for (i, j) in inverse_constr_vio_list_of_end_points:
            print "{},{}".format(i, j)
            print "actual scores are {} and {}".format(point_rel[i][j],point_rel[j][i])
            print "\n"
            print "||||" * 20
        print "======" * 20

        print "opposite relations are assigned in following pairs"
        for (i, j) in opposite_relations_list_of_end_points:
            print "{},{}".format(i, j)
            print "actual scores are {}".format(point_rel[i][j])
            print "\n"
            print "||||" * 20
        print "======" * 20

        print "transitive constraints are not followed in -"
        for (i, j, k) in trans_constr_vio_list_of_end_points:
            print "{},{},{}".format(i, j,k)
            # print "actual scores are {}".format(point_rel[i][j])

            print "i,j score {}".format(point_rel[i][j])
            print "j,k score {}".format(point_rel[j][k])
            print "i,k score {}".format(point_rel[i][k])

            print "\n"
            print "||||" * 20
        print "======" * 20

        return more_than_one_base_rel_list_of_end_points,trans_constr_vio_list_of_end_points,inverse_constr_vio_list_of_end_points,opposite_relations_list_of_end_points



    def heauristics_to_correct_point_relations(self,point_rel,more_than_one_base_rel_list_of_end_points,trans_constr_vio_list_of_end_points,inverse_constr_vio_list_of_end_points,opposite_relations_list_of_end_points):


        a = 0
        b = 1
        id = 2
        leq = 3
        geq = 4


        num_end_points = point_rel.shape[0]

        #dealing with more than one base relation; putting disjunction or removing impossible relation (<,>)




        for i,j in more_than_one_base_rel_list_of_end_points:
            if point_rel[i][j][a] + point_rel[i][j][id]  > 1:
                point_rel[i][j] = point_rel[i][j] * 0
                point_rel[i][j][geq] = 1

            if point_rel[i][j][b] + point_rel[i][j][id]  > 1:
                point_rel[i][j] = point_rel[i][j] * 0
                point_rel[i][j][leq] = 1


            if point_rel[i][j][a] + point_rel[i][j][b]  > 1:
                point_rel[i][j] = point_rel[i][j] * 0





        num_iteration = 0
        while len(trans_constr_vio_list_of_end_points)!=0 and num_iteration<10:

            # print "creating i,j .. j,k .. i,k list"
            i_j_list = {}
            j_k_list = {}
            i_k_list = {}
            for i, j, k in trans_constr_vio_list_of_end_points:
                # print "{},{},{}".format(i, j, k)

                i_j_list[(i, j)] = i_j_list.get((i, j), 0) + 1
                j_k_list[(j, k)] = j_k_list.get((j, k), 0) + 1
                i_k_list[(i, k)] = i_k_list.get((i, k), 0) + 1

            # print "i,j list"
            # for key,value in i_j_list.iteritems():
            #     print "{}   {}".format(key, value)
            #
            # print "-----------------------"
            # print "j,k list"
            # for key, value in j_k_list.iteritems():
            #     print "{}   {}".format(key, value)
            #
            #
            # print "-----------------------"
            # print "i,k list"
            # for key, value in i_k_list.iteritems():
            #     print "{}   {}".format(key,value)

            while len(trans_constr_vio_list_of_end_points)!=0:

                max_i_j = -50
                max_j_k = -50
                max_i_k = -50

                if any(i_j_list):
                    max_i_j = max(i_j_list.values())
                if any(j_k_list):
                    max_j_k = max(j_k_list.values())
                if any(i_k_list):
                    max_i_k = max(i_k_list.values())

                if (max_i_k==max_j_k==max_i_j==-50):
                    print "solved all from i,j; j,k ; i,k dicts but still left some in lists"
                    print trans_constr_vio_list_of_end_points
                    # break
                    sys.exit("exitting")

                # print "max_i_j  ",max_i_j
                # print "max_j_k  ",max_j_k
                # print "max_i_k  ",max_i_k


                if max_i_j>=max_j_k and max_i_j>=max_i_k:

                    # print "in i,j subroutine"

                    i = None
                    j = None
                    k= None

                    universal_point_relations = [0,1,2,3,4]
                    for key,value in i_j_list.iteritems():
                        if value==max_i_j:
                            i,j = key
                            break
                    # print "{},{}".format(i,j)

                    del i_j_list[(i, j)]



                    # print "original relations",point_rel[i][j]
                    if sum(point_rel[i][j])>0:
                        universal_point_relations.remove(np.where(point_rel[i][j]==1)[0][0])
                    # print "universal relation after deleting assigned relation",universal_point_relations
                    point_rel[i][j] = point_rel[i][j] * 0
                    for rel in universal_point_relations:

                        point_rel[i][j][rel] =1

                        correct_rel_found = True
                        for k in range(num_end_points):
                            if i != j != k:
                                if not self.is_transitive_constraint_followed(point_rel[i][j], point_rel[j][k],
                                                                          point_rel[i][k]):
                                    correct_rel_found = False
                                    break

                        if correct_rel_found == True:
                            # print "found correct relation ",rel
                            break

                        else:
                            point_rel[i][j] = point_rel[i][j] * 0

                    for _i,_j,_k in trans_constr_vio_list_of_end_points:
                        if _i == i and _j == j:
                            trans_constr_vio_list_of_end_points.remove((_i,_j,_k))

                elif max_j_k >= max_i_j and  max_j_k >= max_i_k:
                    # print "in j,k subroutine"
                    i = None
                    j = None
                    k= None

                    universal_point_relations = [0, 1, 2, 3, 4]
                    for key, value in j_k_list.iteritems():
                        if value == max_j_k:
                            j,k = key
                            break

                    # print "{},{}".format(j,k)
                    del j_k_list[(j, k)]

                    # print "original relations", point_rel[j][k]

                    if sum(point_rel[j][k]) > 0:
                        universal_point_relations.remove(np.where(point_rel[j][k] == 1)[0][0])
                    # print "universal relation after deleting assigned relation", universal_point_relations
                    point_rel[j][k] = point_rel[j][k] * 0
                    for rel in universal_point_relations:

                        point_rel[j][k][rel] = 1

                        correct_rel_found = True
                        for i in range(num_end_points):
                            if i != j != k:
                                if not self.is_transitive_constraint_followed(point_rel[i][j], point_rel[j][k],
                                                                          point_rel[i][k]):
                                    correct_rel_found = False
                                    break

                        if correct_rel_found == True:
                            # print "found correct relation ", rel
                            break

                        else:
                            point_rel[j][k] = point_rel[j][k] * 0

                    for _i, _j, _k in trans_constr_vio_list_of_end_points:
                        if _k == k and _j == j:
                            trans_constr_vio_list_of_end_points.remove((_i, _j, _k))



                else:

                    # print "in i,k subroutine"
                    i = None
                    j = None
                    k= None

                    universal_point_relations = [0, 1, 2, 3, 4]
                    for key, value in i_k_list.iteritems():
                        if value == max_i_k:
                            i, k = key
                            break

                    # print "{},{}".format(i,k)
                    del i_k_list[(i, k)]

                    # print "original relations", point_rel[i][k]
                    if sum(point_rel[i][k]) > 0:
                        universal_point_relations.remove(np.where(point_rel[i][k] == 1)[0][0])
                    # print "universal relation after deleting assigned relation", universal_point_relations
                    point_rel[i][k] = point_rel[i][k] * 0
                    for rel in universal_point_relations:

                        point_rel[i][k][rel] = 1

                        correct_rel_found = True
                        for j in range(num_end_points):
                            if i != j != k:
                                if not self.is_transitive_constraint_followed(point_rel[i][j], point_rel[j][k],
                                                                          point_rel[i][k]):
                                    correct_rel_found = False
                                    break

                        if correct_rel_found == True:
                            # print "found correct relation ", rel
                            break

                        else:
                            point_rel[i][k] = point_rel[i][k] * 0

                    for _i, _j, _k in trans_constr_vio_list_of_end_points:
                        if _i == i and _k == k:
                            trans_constr_vio_list_of_end_points.remove((_i, _j, _k))
            num_iteration+=1
            print num_iteration
            more_than_one_base_rel_list_of_end_points, trans_constr_vio_list_of_end_points, inverse_constr_vio_list_of_end_points, opposite_relations_list_of_end_points = self.check_constraints_on_end_points(
                point_rel)

        if len(more_than_one_base_rel_list_of_end_points)>0:
            print "more_than_one_base_rel_list_of_end_points"
            print more_than_one_base_rel_list_of_end_points

        if len(trans_constr_vio_list_of_end_points)>0:
            print "trans_constr_vio_list_of_end_points"
            print trans_constr_vio_list_of_end_points

        if len(inverse_constr_vio_list_of_end_points)>0:
            print "inverse_constr_vio_list_of_end_points"
            print inverse_constr_vio_list_of_end_points

        if len(opposite_relations_list_of_end_points)>0:
            print "opposite_relations_list_of_end_points"
            print opposite_relations_list_of_end_points

        # print "while loop completed .... exitting"
        # sys.exit()
        return point_rel

    def keep_only_gold_event_pair_relations(self,rel_pred,event_pairs,events):
        new_rel_pred =[]
        print("predicted relations dimentions :", rel_pred.shape)
        for evs in event_pairs:
            ev1,ev2 = evs.strip().split(feat_separator)
            ev1_ind = events.index(ev1)
            ev2_ind = events.index(ev2)
            new_rel_pred.append(rel_pred[ev1_ind][ev2_ind])
        new_rel_pred = np.array(new_rel_pred)
        print("predicted relations dimentions :", new_rel_pred.shape)
        return np.array(new_rel_pred)



if __name__ == "__main__":
    # num_relations = 13

    num_interval_relations = 6
    num_point_relations = 3
    test_path = ""
    f_list = f_list = ['APW19980227.0487_TD.tml', 'CNN19980223.1130.0960_TD.tml', 'NYT19980212.0019_TD.tml', 'PRI19980216.2000.0170_TD.tml', 'ed980111.1130.0089_TD.tml']
    # mc = maintain_consistency(num_interval_relations,num_base_point_relations,test_path)
    mc = maintain_consistency()
    # mc.get_final_result_report()
    # mc.read_interval_relations_events_and_probability_score(f_list)
    mc.read_end_point_relations_events_and_probability_score(f_list)

    # num_red_relations = 6
    # vec_files_path = "/home/magnet/onkarp/Data/temporal_relations/processed_data/vector_files"
