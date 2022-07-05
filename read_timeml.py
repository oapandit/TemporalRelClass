import os
from xml.dom import minidom
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from shutil import copyfile
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
from constants import *
from utils import *
import re

mirza_data_path = "/home/magnet/onkarp/Data/temporal_relations/processed_data/raw_text/mirza_data"

all_rel_type_counter_dict = {}

class TimeMLReader:
    def __init__(self,source_path,data_set_name = None):
        self.ip_path = source_path
        create_dir(processed_text_data_path)
        self.op_path = processed_text_data_path
        self.num_event_sents_list = []
        self.num_event_doc_list = []

        self.info_tags = ["MAKEINSTANCE","TLINK"]
        self.stopWords = STOPWORDS
        self.rel_type_counter_dict = {}

        self.data_set_extn = "_"+data_set_name if data_set_name is not None else ""

    def read_data_from_all_mls(self):

        f_list = get_list_of_files_with_extn_in_dir(self.ip_path,".tml")
        for f in f_list:
            logger.debug(20*"=====")
            logger.debug("\n File : {}".format(f))
            self.start_of_file = True

            file_name_with_dataset_extn = f[:-4]+self.data_set_extn+".tml"
            xml_doc = minidom.parse(os.path.join(self.ip_path,f))
            self.extract_event_context(xml_doc)
            self.extract_event_linguistic_aspect(xml_doc)
            self.extract_relations(xml_doc)
            self.generate_processed_file(file_name_with_dataset_extn)
        self.generate_reduced_relations_raw_files(self.op_path)

        return self.rel_type_counter_dict

    def extract_event_context(self,xml_doc):

        timeml_root = xml_doc.firstChild
        self.event_id_text_dict = {}
        if len(xml_doc.getElementsByTagName("P"))>0:
            # print("P ROOT FOUND")
            for node in xml_doc.getElementsByTagName("P"):
                self.extract_data(node)
        elif len(timeml_root.getElementsByTagName("TEXT"))>0:
            # print("TEXT ROOT FOUND")
            text_root = timeml_root.getElementsByTagName("TEXT")[0]
            self.extract_data(text_root)
        else:
            # print("TIMEML ROOT ONLY")
            self.extract_data(timeml_root)


    def extract_event_linguistic_aspect(self,xml_doc):

        self.eiid_info_dict ={}
        # instances = xml_doc.findall('MAKEINSTANCE')
        instances = xml_doc.getElementsByTagName('MAKEINSTANCE')
        for instance in instances:
            attrib_dict = {}
            eiid = instance.getAttribute('eiid')
            attrib_dict['eventID'] = instance.getAttribute('eventID')
            attrib_dict['tense'] = instance.getAttribute('tense')
            attrib_dict['pos'] = instance.getAttribute('pos')
            attrib_dict['polarity'] = instance.getAttribute('polarity')
            attrib_dict['aspect'] = instance.getAttribute('aspect')
            self.eiid_info_dict[eiid] = attrib_dict


    def extract_relations(self,xml_doc):
        self.relations_dict_list = []
        # tlinks = xml_doc.findall('TLINK')
        tlinks = xml_doc.getElementsByTagName('TLINK')
        for tlink in tlinks:
            rel_dict = {}
            relation = tlink.getAttribute('relType').upper()
            rel_dict['rel'] = relation

            if tlink.hasAttribute("timeID") and tlink.hasAttribute("relatedToTime"):
                rel_dict['t1'] = tlink.getAttribute('timeID')
                rel_dict['t2'] = tlink.getAttribute('relatedToTime')
            if tlink.hasAttribute("eventInstanceID") and tlink.hasAttribute("relatedToEventInstance"):
                rel_dict['e1'] = tlink.getAttribute('eventInstanceID')
                rel_dict['e2'] = tlink.getAttribute('relatedToEventInstance')
                self.rel_type_counter_dict[relation] = self.rel_type_counter_dict.get(relation,0)+1
                all_rel_type_counter_dict[relation] = all_rel_type_counter_dict.get(relation, 0) + 1
            if tlink.hasAttribute("eventInstanceID") and tlink.hasAttribute("relatedToTime"):
                rel_dict['e1'] = tlink.getAttribute('eventInstanceID')
                rel_dict['t2'] = tlink.getAttribute('relatedToTime')
            if tlink.hasAttribute("timeID") and tlink.hasAttribute("relatedToEventInstance"):
                rel_dict['t1'] = tlink.getAttribute('timeID')
                rel_dict['e2'] = tlink.getAttribute('relatedToEventInstance')

            self.relations_dict_list.append(rel_dict)

    def extract_data(self,timeml_root):
        sent = ""
        event_number = 0
        sent_number = 0
        for index,node in enumerate(timeml_root.childNodes):
            # if node is not None:
                nodeType,data = self.get_data(node)
                if self.start_of_file:
                    self.start_of_file = False
                    # print([data])
                    if "_" in data:
                        data = data.split('_')[1]

                    if "--" in data:
                        data = data.split('--')[1]

                    data = data.split('\n\n\n\t')[0]
                    data = data.split('\n\n\t')[0]
                    data = data.split('\n\n\n\n')[0]

                    # print("data after change")
                    # print([data])

                sent += " "+data
                # logger.debug("\n ****** \n"
                # print(sent)

                # logger.debug("Nodetype : ", nodeType
                # logger.debug("Data : ",data
                if nodeType in self.info_tags:
                    # print("END of Text")
                    break
                if nodeType =="EVENT": #or nodeType =="TIMEX3" : IF we want to consider timex data
                    # logger.debug("Adding entry to dictionary",num_event_sent
                    event_number +=1
                    self.add_entry_to_dict(timeml_root,sent,index,event_number,sent_number)
                if nodeType == "DOC" or nodeType =="TIMEX3": # remove or nodeType =="TIMEX3" if want to consider timex data
                    continue
                else:
                    sent = sent.strip()
                    sent_splitter = sent_tokenize(sent)

                    if len(sent_splitter)>0:
                        sent = sent_splitter[-1]
                    if len(sent_splitter)>1:
                        # print("LENGTH AFTER SENTENCE SPLITTING", len(sent_splitter))
                        # self.num_event_sents_list.append(num_event_sent)
                        sent_number += 1


    def add_entry_to_dict(self,timeml_root,c_sent,c_index,event_number,sent_number):

        node = timeml_root.childNodes[c_index]
        tense_indic_dict = {"'s":'is', "'re":'are', "'ll":"will", "'ve":'have',"'m":'am'}

        def _filter_words(words_to_filter):
            words_without_punct = [w for w in words_to_filter if w not in string.punctuation]
            if len(words_without_punct)>0:
                # ev_word = words_without_punct[-1]
                logger.debug("words_without_punct : {}".format(words_without_punct))
                words_without_punct = words_without_punct[:-1]
                words_without_punct = [w for w in words_without_punct if w not in self.stopWords]

                for ind,w in enumerate(words_without_punct):
                    if tense_indic_dict.get(w,None) is not None:
                        words_without_punct[ind] = tense_indic_dict[w]
                # words_without_punct.append(ev_word)
            return words_without_punct


        def __adjust_left_context_num_words(words_without_punct):
            cont = ""
            if len(words_without_punct)>=WORD_CONTEXT_NUM +1:
                cont = ' '.join(words_without_punct[len(words_without_punct)-WORD_CONTEXT_NUM-1:])
            else:
                number_words_pad = WORD_CONTEXT_NUM - len(words_without_punct) +1
                while number_words_pad>0:
                    cont += " "
                    cont +=PAD_WORD
                    number_words_pad -=1
                cont += ' '+' '.join(words_without_punct)
            return cont

        def __adjust_right_context_num_words(right_context,cont):
            while right_context < WORD_CONTEXT_NUM:
                cont += ' ' + PAD_WORD
                right_context += 1
            return cont


        def _get_context_for_event(index,sent):
            logger.debug(20 * "=====")
            logger.debug("Sentence given to the event function : {}".format(sent))
            words = word_tokenize(sent)
            logger.debug(words)
            words_without_punct = _filter_words(words)
            logger.debug(words_without_punct)
            logger.debug(4*"&&&")
            logger.debug( "Number of words for event : {} ... printing all words --".format(len(words_without_punct)))
            for w in words_without_punct:
                logger.debug(w)

            cont = __adjust_left_context_num_words(words_without_punct)

            index +=1
            right_context = 0
            while right_context<WORD_CONTEXT_NUM and index < len(timeml_root.childNodes):

                nodeType,data = self.get_data(timeml_root.childNodes[index])
                logger.debug("Data while getting right context : {}".format(data))
                if (data == " " or data is None or data == "\n" or data == "" or data == " \n") and nodeType not in self.info_tags:
                    index += 1
                    continue
                if nodeType in self.info_tags:
                    break
                else:
                    sent_splitter = sent_tokenize(data)
                    sent = sent_splitter[0]
                    words = word_tokenize(sent)
                    words_without_punct = _filter_words(words)
                    if len(words_without_punct) + right_context >= WORD_CONTEXT_NUM :
                        cont += ' '+' '.join(words_without_punct[: WORD_CONTEXT_NUM -right_context])
                    else:
                        cont += ' '+' '.join(words_without_punct)
                    right_context += len(words_without_punct)

                    if len(sent_splitter) > 1:
                        break

                index+=1


            cont = __adjust_right_context_num_words(right_context, cont)
            logger.debug("Context : {}".format(cont))
            return cont


        def _get_complete_sentence(index,sent):
            while index < len(timeml_root.childNodes):
                nodeType,data = self.get_data(timeml_root.childNodes[index])
                logger.debug("Data while getting right context : {}".format(data))
                if (data == " " or data is None or data == "\n" or data == "" or data == " \n") and nodeType not in self.info_tags:
                    index += 1
                    continue
                if nodeType in self.info_tags:
                    break
                else:
                    sent_splitter = sent_tokenize(data)
                    sent = sent + " "+ sent_splitter[0]
                    if len(sent_splitter) > 1:
                        break
                index+=1
            sent = " ".join(sent.split())
            # if "_" in sent:
            #     print(sent)
            return sent

        id = node.getAttribute('eid')

        words_without_punct = [w for w in c_sent.split() if w not in string.punctuation]
        position_event_in_sentence = len(words_without_punct)-1


        context = _get_context_for_event(c_index,c_sent)
        sentence = _get_complete_sentence(c_index+1,c_sent)

        # print("Event : ",context.split()[4])


        if position_event_in_sentence > 100:
            print("Received sentence : ", c_sent)
            print("position of event ", position_event_in_sentence)
            print("Complete sentence : ", sentence)
        if len(context.split()) != 2*WORD_CONTEXT_NUM +1:
            logger.debug("###ERROR ####")
            logger.debug("Context : ",context)
            logger.debug("#####")


        event_info_dict = {}
        event_info_dict['EVENT_WORD'] = context.split()[4]
        event_info_dict['CONTEXT'] = context
        event_info_dict['EVENT_NUM'] = str(event_number)
        event_info_dict['SENTENCE_NUM'] = str(sent_number)
        event_info_dict['SENTENCE'] = sentence
        event_info_dict['EVENT_POSITION'] = str(position_event_in_sentence)
        self.event_id_text_dict[id] = event_info_dict
        # logger.debug(20*"##"


    def get_data(self,node):

        nodeType = ""
        data = ""
        # print(node.nodeName)
        if node.nodeName == '#text':
            data = node.data
            nodeType = "TEXT"
        elif node.nodeName not in self.info_tags:
            # print("elif loop")
            try:
                data = node.firstChild.data
                nodeType = node.nodeName
            except AttributeError:
                pass
                # logger.debug(" ERROR : No data"
        else:
            nodeType = node.nodeName
        data = data.strip()
        # print(data)
        return nodeType,data

    def generate_processed_file(self,filename):
        neglect_this_rels = ['BEGINS', 'ENDS', 'ENDED_BY', 'BEGUN_BY', 'IBEFORE', 'IAFTER', 'OVERLAP']

        def _get_feat_from_dicts(eiid):
            event_attrib_dict = self.eiid_info_dict.get(eiid,None)
            if event_attrib_dict is not None:
                event_id = event_attrib_dict.get('eventID')
                event_info_dict = self.event_id_text_dict[event_id]
                features = []
                for feat in ling_aspects:
                    features.append(event_attrib_dict.get(feat))
                for feat in event_context_feat:
                    features.append(event_info_dict.get(feat))
                event_feats = feat_separator.join(features)
                return event_feats
            else:
                print("There is no event eiid :  , returning null features".format(eiid))

        def _create_event_file():
            for eiid in sorted(self.eiid_info_dict.keys()):
                event_feat = _get_feat_from_dicts(eiid)
                event_feat = eiid+feat_separator+event_feat
                event_text_file.write(event_feat + "\n")
            event_text_file.close()



        feat_separator = "$#$#$#$"
        ling_aspects = ['tense','pos','polarity','aspect']
        event_context_feat = ['EVENT_WORD','CONTEXT','EVENT_NUM','SENTENCE_NUM','SENTENCE','EVENT_POSITION']

        event_text_file = open(os.path.join(self.op_path, filename[:-4] + ".events"), 'w')
        event1_text_file = open(os.path.join(self.op_path, filename[:-4]+".ev1"), 'w')
        event2_text_file = open(os.path.join(self.op_path, filename[:-4] + ".ev2"), 'w')
        rel_text_file = open(os.path.join(self.op_path, filename[:-4] + ".rel"), 'w')
        event_pair_id_file = open(os.path.join(self.op_path, filename[:-4] + ".ev_pairs"), 'w')

        _create_event_file()

        for rel_dict in self.relations_dict_list:
            if rel_dict.get('e1',None) is not None and rel_dict.get('e2',None) is not None:
                if rel_dict.get('rel') not in neglect_this_rels:
                    e1_iid = rel_dict.get('e1')
                    e2_iid = rel_dict.get('e2')

                    e1_feats = _get_feat_from_dicts(e1_iid)
                    e2_feats = _get_feat_from_dicts(e2_iid)

                    if e1_feats is not None and e2_feats is not None:

                        event1_text_file.write(e1_feats + "\n")
                        event2_text_file.write(e2_feats + "\n")
                        rel_text_file.write(rel_dict.get('rel') + "\n")
                        event_pair_id_file.write(e1_iid+feat_separator+e2_iid + "\n")


        event1_text_file.close()
        event2_text_file.close()
        rel_text_file.close()
        event_pair_id_file.close()

    def generate_reduced_relations_raw_files(self, lim_rel_dir):

        if not os.path.exists(lim_rel_dir):
            os.makedirs(lim_rel_dir)

        rel_files_list = get_list_of_files_with_extn_in_dir(self.op_path,".rel")

        neglect_this_rels = ['BEGINS','ENDS','ENDED_BY','BEGUN_BY','IBEFORE','IAFTER','OVERLAP']

        for rel_f in rel_files_list:
            ev1_f = rel_f[:-3]+ "ev1"
            ev2_f = rel_f[:-3]+ "ev2"

            with open(os.path.join(self.op_path, ev1_f), "r") as f:
                ev1_raw_data = f.readlines()

            with open(os.path.join(self.op_path, ev2_f), "r") as f:
                ev2_raw_data = f.readlines()

            with open(os.path.join(self.op_path, rel_f), "r") as f:
                rel_raw_data = f.readlines()


            event1_lim_text_file = open(os.path.join(lim_rel_dir, ev1_f), 'w')
            event2_lim_text_file = open(os.path.join(lim_rel_dir, ev2_f), 'w')
            rel_lim_text_file = open(os.path.join(lim_rel_dir, rel_f), 'w')


            for i,line in enumerate(rel_raw_data):
                rel = line.strip().upper()
                if rel in neglect_this_rels:
                    continue
                else:
                    event1_lim_text_file.write(ev1_raw_data[i])
                    event2_lim_text_file.write(ev2_raw_data[i])
                    rel_lim_text_file.write(line)


            event1_lim_text_file.close()
            event2_lim_text_file.close()
            rel_lim_text_file.close()


if __name__ == '__main__':


    time_bank_path = os.path.join(raw_data_path,"TimeBank")
    vc_raw_data_path = os.path.join(raw_data_path,"vc_dataset")
    td_raw_data_path = os.path.join(raw_data_path,"td_dataset")
    aq_raw_data_path = os.path.join(raw_data_path,"AQUAINT")
    te3_pt_raw_data_path = os.path.join(raw_data_path,"te3_platinum")

    raw_data_paths = [aq_raw_data_path,time_bank_path,vc_raw_data_path,td_raw_data_path,te3_pt_raw_data_path]
    data_sets_name = ["AQUAINT","TIMEML","VC","TIMEDENSE","TE3_PT"]
    data_sets_extn = ["AQ", "TB", None, None, "TE3PT"]

    for i,r_path in enumerate(raw_data_paths):
        logger.debug(20 * "========")
        logger.debug("PROCESSING DATASET : "+data_sets_name[i])
        logger.debug(20 * "========")
        reader = TimeMLReader(r_path,data_sets_extn[i])
        rel_type_counter_dict = reader.read_data_from_all_mls()

    logger.debug(10*"=====")
    logger.debug("completion of processed text file generation.")
