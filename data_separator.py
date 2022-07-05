import os
from constants import *


class data_separator:
    '''
        Train data : TB +AQ + VC + TD-train
        Development data : TD-Dev
        Test data : TD-test + TE3-pt

        currently we have not processed TE3-pt data, so we are just segragating TD data. All other is training data by itself.

    '''

    def __init__(self,vec_file_path):
        self.vec_file_path = vec_file_path
        self.train_folder = os.path.join(self.vec_file_path,"train_data")
        self.dev_folder = os.path.join(self.vec_file_path, "dev_data")
        self.test_folder = os.path.join(self.vec_file_path, "test_data")
        # self.vec_file_path = self.train_folder
        self.TD_dev_docs = [
                            "APW19980227.0487.tml",
                            "CNN19980223.1130.0960.tml",
                            "NYT19980212.0019.tml",
                            "PRI19980216.2000.0170.tml",
                            "ed980111.1130.0089.tml"
                        ]

        # self.TD_dev_docs = ['AP_20130322.tml', 'CNN_20130321_821.tml', 'CNN_20130322_1003.tml', 'CNN_20130322_1243.tml', 'CNN_20130322_248.tml', 'CNN_20130322_314.tml', 'WSJ_20130318_731.tml', 'WSJ_20130321_1145.tml', 'WSJ_20130322_159.tml', 'WSJ_20130322_804.tml', 'bbc_20130322_1150.tml', 'bbc_20130322_1353.tml', 'bbc_20130322_1600.tml', 'bbc_20130322_332.tml', 'bbc_20130322_721.tml', 'nyt_20130321_china_pollution.tml', 'nyt_20130321_cyprus.tml', 'nyt_20130321_sarkozy.tml', 'nyt_20130321_women_senate.tml', 'nyt_20130322_strange_computer.tml']
        self.TD_test_docs = [
            "APW19980227.0489.tml",
            "APW19980227.0494.tml",
            "APW19980308.0201.tml",
            "APW19980418.0210.tml",
            "CNN19980126.1600.1104.tml",
            "CNN19980213.2130.0155.tml",
            "NYT19980402.0453.tml",
            "PRI19980115.2000.0186.tml",
            "PRI19980306.2000.1675.tml" ]

        self.te3pt_extn = "_TE3PT"

        if not os.path.exists(self.train_folder):
            os.makedirs(self.train_folder)

        if not os.path.exists(self.test_folder):
            os.makedirs(self.test_folder)

        if not os.path.exists(self.dev_folder):
            os.makedirs(self.dev_folder)


    @property
    def test_folder_path(self):
        return self.test_folder


    @property
    def dev_folder_path(self):
        return self.dev_folder

    def _get_list_of_files(self, file_path):
        filelist = [name for name in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, name))]
        filelist.sort()
        return filelist

    def segrate_docs(self):
        file_list = self._get_list_of_files(self.vec_file_path)

        TD_dev_substrings =[]
        TD_test_substrings = []

        for doc in self.TD_dev_docs:
            TD_dev_substrings.append(doc[:-4]+"_TD")
            # TD_dev_substrings.append(doc[:-4])

        for doc in self.TD_test_docs:
            TD_test_substrings.append(doc[:-4]+"_TD")



        for f in file_list:

            f_substring = f[:f.rindex('.')]

            if f_substring in TD_dev_substrings:
                os.rename(os.path.join(self.vec_file_path,f), os.path.join(self.dev_folder,f))

            elif f_substring in TD_test_substrings or self.te3pt_extn in f:
                os.rename(os.path.join(self.vec_file_path, f), os.path.join(self.test_folder, f))

            else:
                os.rename(os.path.join(self.vec_file_path, f), os.path.join(self.train_folder, f))



if __name__ == "__main__":

    ds = data_separator(processed_vec_data_path)
    ds.segrate_docs()
