import os
from utils import *
from lxml import etree as ET
import logging
from apetite.TimeMLDocument import Document
import time
import shutil


logging.basicConfig(filename="tml_process.log",level=logging.INFO,format="%(levelname)s:%(message)s")

sss = ['BEFOREI', 'IS_INCLUDED', 'STARTI', 'DURING', 'IBEFORE', 'SIMULTANEOUS',
       'IAFTER', 'MEET', 'MEETI', 'VAGUE', 'FINISHI', 'ENDS', 'FINISH', 'EQUALS',
       'OVERLAP', 'INCLUDES', 'START', 'DURINGI', 'OVERLAPI', 'IDENTITY', 'BEFORE',
       'BEGINS', 'BEGUN_BY', 'AFTER', 'ENDED_BY']
rel_dict={"beforei":"after","is_included":"during","includes":"during_inv","duringi":"during_inv",
          "identity":"simultaneous","equals":"simultaneous"}

ignore_rels = ['STARTI','IBEFORE', 'IAFTER', 'MEET', 'MEETI', 'FINISHI', 'ENDS', 'FINISH','OVERLAP',
               'START', 'OVERLAPI','BEGINS', 'BEGUN_BY', 'ENDED_BY']



class TMLStats:

    def __init__(self,tml_files_path):
        self.tml_files_path = tml_files_path
        self.tml_files = get_list_of_files_with_extn_in_dir(self.tml_files_path,".tml")
        self.file_tlink_event_dict = {}


    def calculate_event_tlinks_stats(self,tml_file_name):

        event_counter = 0.0
        tlink_counter = 0

        logging.debug(10*"-----")
        tml_f_path = os.path.join(self.tml_files_path, tml_file_name)
        logging.debug("parsing file {}".format(tml_f_path))

        tml_cont = ET.parse(tml_f_path)
        tml_root = tml_cont.getroot()

        tlinks = tml_cont.findall('TLINK')
        event_instances = tml_cont.findall('MAKEINSTANCE')

        # event_counter = len(event_instances)
        logging.debug("number of MAKEINSTANCE in file {}".format(len(event_instances)))

        for inst in event_instances:
            if inst.attrib.get('eventID',None) is not None:
                event_counter +=1

        logging.debug("number of event ids {}".format(event_counter))

        if event_counter!=len(event_instances):
            logging.error("number of event ids {} and number of event instances {} don't match".format(event_counter,len(event_instances)))

        logging.debug("tlinks contain relations between timex as well as events, we want only events related tlinks. so calcualting it.")
        tlink_dict = {}
        for tlink in tlinks:
            if tlink.get("eventInstanceID", None) is not None and tlink.get("relatedToEventInstance", None) is not None:
                e1 = tlink.get("eventInstanceID", None)
                e2 = tlink.get("relatedToEventInstance", None)
                if tlink_dict.get((e1,e2),None) is None:
                    tlink_dict[(e1,e2)] = 1
                    tlink_dict[(e2, e1)] = 1
                    tlink_counter+=2
                # tlink_counter += 1

        logging.debug("tlinks between events is {}".format(tlink_counter))

        expected_num_tlinks = (event_counter*(event_counter-1)/2)
        percentage = float(tlink_counter/expected_num_tlinks)*100

        logging.debug("Expected {} event-tlinks in file but only {} are present. Percentage is {} %.".format(expected_num_tlinks,tlink_counter,percentage))

        self.file_tlink_event_dict[tml_file_name] = [event_counter,tlink_counter,expected_num_tlinks]


    def get_ev_tlink_stats_for_all_files(self):

        for tml_file in self.tml_files:
            self.calculate_event_tlinks_stats(tml_file)

        event_counter = 0.0
        tlink_counter = 0
        expected_num_tlinks = 0

        for k,v in self.file_tlink_event_dict.items():
            event_counter = event_counter + v[0]
            tlink_counter = tlink_counter + v[1]
            expected_num_tlinks = expected_num_tlinks + v[2]

        percentage = float(tlink_counter/expected_num_tlinks)*100

        logging.info(10*"####")
        logging.info("Over all files in the path <<{}>> expected event-tlinks are {} but only {} are present. Percentage is {} %.".format(self.tml_files_path,expected_num_tlinks,tlink_counter,percentage))

        return event_counter,tlink_counter,expected_num_tlinks


    def get_all_the_reltypes(self,files_path):

        logging.info("searching for all tml files in path {}".format(files_path))
        tml_file_list = search_all_file_in_dir(files_path,"*.tml")

        num_tml_files = len(tml_file_list)
        logging.info("number of tml files {} ".format(num_tml_files))
        rels = []
        rels_freq_dict = {}
        for tml_file in tml_file_list:
            logging.debug(5*"---")
            logging.debug("reading file {}".format(tml_file))

            tml_cont = ET.parse(tml_file)
            tlinks = tml_cont.findall('TLINK')

            for tlink in tlinks:
                if tlink.get("eventInstanceID", None) is not None and tlink.get("relatedToEventInstance",None) is not None:
                    relation = tlink.attrib["relType"]
                    rels.append(relation)
                    rels_freq_dict[relation] = rels_freq_dict.get(relation,0) +1


        logging.info(set(rels))
        to_be_deleted_links = 0
        total_tlinks = 0
        for k,v in rels_freq_dict.items():
            if k in ignore_rels:
                to_be_deleted_links = to_be_deleted_links + v
            total_tlinks = total_tlinks + v
            logging.info("{} : {}".format(k,v))

        logging.info("total tlinks {} of which will be ignored {} so remaining {}".format(total_tlinks,to_be_deleted_links,total_tlinks-to_be_deleted_links))


class TMLProcess:

    def __init__(self,dataset_path):
        self.dataset_path = dataset_path

    def check_if_consistent(self):

        logging.info("searching for all tml files in path {}".format(self.dataset_path))
        tml_file_list = search_all_file_in_dir(self.dataset_path,"*.tml")

        num_tml_files = len(tml_file_list)
        logging.info("number of tml files {} ".format(num_tml_files))
        num_inconsistency = 0
        # tml_file_list= ["/home/magnet/onkarp/Data/temporal_relations/raw_data/td_dataset/PRI19980121.2000.2591_TD.tml"]
        for tml_file in tml_file_list:
            logging.info(5*"---")
            logging.info("checking consistency of file {}".format(tml_file))
            temp_graph = Document(tml_file).get_graph()
            t0 = time.time()
            is_consistent = temp_graph.saturate()
            t1 = time.time()
            logging.info("saturation of graph in {0}s".format((t1 - t0)))
            if not is_consistent:
                logging.info("File : {0} is inconsistent.".format(tml_file))
                num_inconsistency+=1
            else:
                logging.info("File : {0} is consistent.".format(tml_file))

        logging.info("Out of {} tml files, {} are inconsistent, i.e. {}% inconsistency".format(num_tml_files,num_inconsistency,num_inconsistency*100/float(num_tml_files)))

    def create_satuarated_files(self,dataset_path=None,in_place=True):

        if not in_place:
            sat_tml_dir = os.path.join(dataset_path,"saturated")
            logging.info("creating saturated dataset directory {}".format(sat_tml_dir))
            create_dir(sat_tml_dir)


        logging.info("searching for all tml files in path {}".format(self.dataset_path))
        tml_file_list = search_all_file_in_dir(self.dataset_path,"*.tml")

        num_tml_files = len(tml_file_list)
        logging.info("number of tml files {} ".format(num_tml_files))

        num_tml_files = len(tml_file_list)
        num_file_saturated = 0

        for tml_file in tml_file_list:
            logging.info(10*"-----")
            logging.info("saturating file {}".format(tml_file))

            if not in_place:
                tml_file_name = os.path.basename(tml_file)
                path, dirname = os.path.split(os.path.dirname(tml_file))
                dest_dir = os.path.join(sat_tml_dir,dirname)
                create_dir(dest_dir)
                # logging.debug("saturating file {}".format(tml_file))
                dest_file_path = os.path.join(dest_dir,tml_file_name)

            else:
                dest_file_path = tml_file


            logging.info("destination file path {}".format(dest_file_path))

            if not os.path.exists(dest_file_path) or in_place:
                temp_graph = Document(tml_file)
                new_rel_dict = temp_graph.get_derived_relations_if_consistent()
                if new_rel_dict is not None:
                    logging.info("creating saturated file for file {} ".format(tml_file))
                    num_file_saturated +=1
                    event_pair_rel_dict = {}
                    for k,v in new_rel_dict.items():
                        logging.debug("k is {} and v is {}".format(k,v))
                        if len(v) ==1:
                            logging.debug("as length of v is 1, considering for sat file.")
                            e1,e2 = k[0].lower(),k[1].lower()
                            rel = []
                            for x in v:
                                rel.append(x)
                            if len(rel) ==1:
                                rel = rel[0]

                            logging.debug("rel between events pair {} {} is {}. considering rel {}.".format(e1,e2,rel,rel[5:].upper()))
                            event_pair_rel_dict[(e1,e2)] = rel[5:].upper() # relation contains suffix as allen_, so neglecting that
                        else:
                            logging.debug("as length of v is more than 1 , not considering for sat file.")

                    logging.debug("creating file  {} ".format(dest_file_path))
                    self._create_file_with_additional_tlinks(tml_file, dest_file_path, event_pair_rel_dict)
                    logging.info("saturated file created {} ".format(dest_file_path))
                else:
                    logging.info("saturated file is not created for file {} ".format(tml_file))

            else:
                logging.info("file already exists not created.")

        logging.info("Total tml files : {} \n Saturated files : {}".format(num_tml_files,num_file_saturated))

    def _create_file_with_additional_tlinks(self, source_file_path, dest_file_path, event_pair_rel_dict):

        newline_char = "\n"

        logging.info("parsing source file {}".format(source_file_path))
        source_tml = ET.parse(source_file_path)
        root = source_tml.getroot()

        tlinks = source_tml.findall('TLINK')

        lid = []
        for tlink in tlinks:
            lid.append(int(tlink.attrib["lid"][1:]))

        lid = max(lid) + 1
        logging.info("number of tlinks in file are {} and highest lid {}".format(len(tlinks),lid-1))

        logging.debug("adding only event-tlinks")
        for k, v in event_pair_rel_dict.items():
            e1,e2 = k
            rel = v
            logging.debug(5*"---")
            logging.debug("event1 {} and event2 {}, relation between them {}".format(e1,e2,rel))
            if "ei" in e1 and "ei" in e2:
                logging.debug("rel is added.")
                tlink = ET.Element("TLINK")
                tlink.set("lid", "l" + str(lid))
                tlink.set("relType", rel)
                tlink.set("eventInstanceID", e1)
                tlink.set("relatedToEventInstance",e2)
                tlink.tail = newline_char
                root.append(tlink)
                lid += 1

            else:
                logging.debug("rel not added.")

        logging.debug("writing xml file")
        source_tml.write(dest_file_path, pretty_print=True)
        logging.debug("file written, returning.")

    def copy_nonsat_files_to_sat_location(self):
        dataset_path = "/home/opandit/Downloads/TML_datasets"
        sat_dataset_path = "/home/opandit/Downloads/TML_datasets/saturated"
        tml_file_list = search_all_file_in_dir(dataset_path, "*.tml")

        for tml_file_source_path in tml_file_list:
            tml_file_name = os.path.basename(tml_file_source_path)
            _, dirname = os.path.split(os.path.dirname(tml_file_source_path))
            tml_file_dest_path = os.path.join(sat_dataset_path, dirname, tml_file_name)

            if not os.path.exists(tml_file_dest_path):
                shutil.copy2(tml_file_source_path, tml_file_dest_path)



    def normalize_rels(self):
        logging.info("searching for all tml files in path {}".format(self.dataset_path))
        tml_file_list = search_all_file_in_dir(self.dataset_path,"*.tml")

        num_tml_files = len(tml_file_list)
        logging.info("number of tml files {} ".format(num_tml_files))

        for tml_file in tml_file_list:
            logging.info(10*"-----")
            logging.info("processing file {}".format(tml_file))

            logging.debug("parsing source file {}".format(tml_file))
            _tml = ET.parse(tml_file)
            _tml_root = _tml.getroot()

            tlinks = _tml.findall('TLINK')

            for tlink in tlinks:
                _tml_root.remove(tlink)

            logging.debug("deleted all tlinks")
            for tlink in tlinks:
                rel = tlink.attrib['relType']
                logging.debug(5*"----")
                logging.debug("relation {}".format(rel))
                if rel.upper() in ignore_rels:
                    logging.debug("ignored")
                    continue
                if rel_dict.get(rel.lower(),None) is not None:
                    logging.debug("modified")
                    reset_rel =rel_dict[rel.lower()]
                    tlink.set("relType", reset_rel.upper())

                _tml_root.append(tlink)


            _tml.write(tml_file)
            logging.debug("file processed.")



if __name__ == '__main__':

    # base_paths = ["/home/opandit/Downloads/TML_datasets","/home/opandit/Downloads/TML_datasets/datasetBCK"]

    base_paths = ["/home/magnet/onkarp/Data/temporal_relations/raw_data"]
    datasets = ["td_dataset","vc_dataset","AQUAINT","TimeBank"]

    # for base_path in base_paths:
    #     glob_event_counter, glob_tlink_counter, glob_expected_num_tlinks = 0, 0, 0
    #     for dataset in datasets:
    #         tml_path = os.path.join(base_path,dataset)
    #         tp = TMLStats(tml_path)
    #         event_counter, tlink_counter, expected_num_tlinks = tp.get_ev_tlink_stats_for_all_files()
    #
    #         glob_event_counter = glob_event_counter + event_counter
    #         glob_tlink_counter = glob_tlink_counter + tlink_counter
    #         glob_expected_num_tlinks = glob_expected_num_tlinks + expected_num_tlinks
    #
    #     percentage = float(glob_tlink_counter/glob_expected_num_tlinks)*100
    #
    #
    #     logging.info(10*"####$$$$####")
    #     logging.info("Over all datasets, expected event-tlinks are {} but only {} are present. Percentage is {} %.".format(glob_expected_num_tlinks,glob_tlink_counter,percentage))
    #
    # tml_path = "/home/magnet/onkarp/Data/temporal_relations/raw_data"
    # tp = TMLStats(tml_path)
    # tp.get_all_the_reltypes(tml_path)
    #
    # tms = TMLProcess(tml_path)
    # tms.normalize_rels()
    #
    # tp.get_all_the_reltypes(tml_path)


    base_path = base_paths[0]
    for dataset in datasets:
        tml_path = os.path.join(base_path, dataset)
        # tms = TMLProcess(tml_path)
        # tms.create_satuarated_files()
        # tms.normalize_rels()
        tp = TMLStats(tml_path)
        tp.get_all_the_reltypes(tml_path)

