import os
from xml.dom import minidom
# import xml.etree.ElementTree as ET
from lxml import etree as ET
from constants import *
from utils import *


class generate_augmented_dataset:
    def __init__(self,timeml_path,aug_file_path,aug_data_path,aug_data_name):
        self.timeml_path = timeml_path
        self.aug_file_path = aug_file_path
        self.aug_data_path = aug_data_path
        self.aug_data_name = aug_data_name
        if not os.path.exists(self.aug_data_path):
            os.makedirs(self.aug_data_path)


    def generate_aug_file(self):


        logger.debug("Generating augmented files for dataset {} at location {}".format(self.aug_data_name,self.aug_data_path))

        newline_char = "\n"

        with open(self.aug_file_path,"r") as f:
            content = f.readlines()

        logger.debug("Augment information text file {} reading completed. number of lines {} ".format(self.aug_file_path,len(content)))
        i=0
        num_errors = 0
        while i < len(content):
            logger.debug("index {}".format(i))
            line = content[i]
            timeml_fname = line.strip().split()[0]+timeml_file_extn
            logger.debug("timeml file  {}".format(timeml_fname))
            current_timeml_fname = timeml_fname

            dest_aug_ml_fname = line.strip().split()[0]+"_"+self.aug_data_name+timeml_file_extn

            try:
                _xmldoc = ET.parse(os.path.join(self.timeml_path,timeml_fname))
            except:
                break
            aug_f_path = os.path.join(self.aug_data_path,dest_aug_ml_fname)
            logger.debug("writing relations at {}".format(aug_f_path))

            event_instances = _xmldoc.findall('MAKEINSTANCE')
            event_dict = {}
            for inst in event_instances:
                ev_id = inst.attrib['eventID']
                eiid = inst.attrib['eiid']
                event_dict[ev_id] = eiid

            root = _xmldoc.getroot()
            # for key, value in event_dict.iteritems():
            #     logger.debug(key,value)

            tlinks = _xmldoc.findall('TLINK')

            lid = get_last_lid(tlinks) +1
            logger.debug("Starting to create new relations with lid {}".format(lid))

            while i<len(content) and current_timeml_fname == timeml_fname:
                is_error= False
                split_cont = line.strip().split()
                logger.debug("list length : {}".format(len(split_cont)))
                logger.debug("relation : {}".format(split_cont[3]))
                logger.debug("event1 : {}".format(split_cont[1]))
                logger.debug("event2 : {}".format(split_cont[2]))
                relation = split_cont[3]
                if not len(relation) >2:
                    relation = rel_dict[relation]
                #TODO : add timex related relations as well
                if "t" not in split_cont[1] and "t" not in split_cont[2]:
                    if "ei" in split_cont[1]:
                        ev1_eiid = split_cont[1]
                    else:
                        if event_dict.get(split_cont[1],None) is None:
                            is_error = True
                            logger.error("error occured, event1 not in dicts {}".format(split_cont[1]))
                        else:
                            ev1_eiid = event_dict[split_cont[1]]
                    if "ei" in split_cont[2]:
                        ev2_eiid = split_cont[2]
                    else:
                        if event_dict.get(split_cont[2],None) is None:
                            is_error = True
                            logger.error("error occured, event2 not in dicts {}".format(split_cont[2]))
                        else:
                            ev2_eiid = event_dict[split_cont[2]]

                    if not is_error:

                        ev1_eiid = str(ev1_eiid)
                        ev2_eiid = str(ev2_eiid)
                        logger.debug("event1 : {}".format(ev1_eiid))
                        logger.debug("event2 :{}".format(ev2_eiid))

                        tlink = ET.Element("TLINK")
                        tlink.set("lid", "l" + str(lid))
                        tlink.set("relType", relation)
                        tlink.set("eventInstanceID", ev1_eiid)
                        tlink.set("relatedToEventInstance", ev2_eiid)
                        tlink.tail = newline_char
                        root.append(tlink)
                        lid+=1
                    else:
                        num_errors+=1
                i+=1
                if i>=len(content):
                    logger.debug("index before breaking {}".format(i))
                    logger.debug("breaking")
                    break
                line = content[i]
                timeml_fname = line.strip().split()[0] + timeml_file_extn
                logger.debug("line : {}".format(line))

            _xmldoc.write(aug_f_path, pretty_print=True)

        logger.debug("Generation of augment file completed. Number of errors {}".format(num_errors))




    def get_unique_relations(self):

        rel_dict = []

        with open(self.aug_file_path,"r") as f:
            content = f.readlines()

        i=0
        while i < len(content):
            line = content[i]
            relation = line.strip().split()[3]
            if relation not in rel_dict:
                rel_dict.append(relation)
            i+=1

        logger.debug("No. of relations {}".format(len(rel_dict)))
        logger.debug("Relations : {}".format(rel_dict))




if __name__ == "__main__":

    time_bank_path = os.path.join(raw_data_path,"TimeBank")
    vc_aug_dataset_path = os.path.join(raw_data_path,"vc_dataset")
    td_aug_dataset_path = os.path.join(raw_data_path,"td_dataset")
    vc_aug_file_path = os.path.join(raw_data_path,"verb_clause.txt")
    td_aug_file_path = os.path.join(raw_data_path,"time_bank_dense.txt")


    gad = generate_augmented_dataset(time_bank_path,td_aug_file_path,td_aug_dataset_path,aug_data_name="TD")
    gad.generate_aug_file()
    # gad.get_unique_relations()


    gad = generate_augmented_dataset(time_bank_path,vc_aug_file_path,vc_aug_dataset_path,aug_data_name="VC")
    gad.generate_aug_file()
    # gad.get_unique_relations()