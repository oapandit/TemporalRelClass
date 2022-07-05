import os

after = 0
before = 1
equal = 2
leq = 3
geq = 4

RELATIONS_DICT = {'AFTER':0, 'BEFORE':1, 'EQUAL':2, 'LEQ':3, 'GEQ':4}

r_dict = {0:'>',1:'<',2:'=',3:'<=',4:">="}
RELATIONS_COMPOSITION = {

    ('AFTER','AFTER'):'AFTER',
    ('AFTER','GEQ'):'AFTER',
    ('AFTER','EQUAL'):'AFTER',

    ('BEFORE','BEFORE'):'BEFORE',
    ('BEFORE','LEQ'):'LEQ',
    ('BEFORE','EQUAL'):'BEFORE',

    ('GEQ', 'AFTER'): 'AFTER',
    ('GEQ', 'GEQ'): 'GEQ',
    ('GEQ', 'EQUAL'): 'EQUAL',

    ('LEQ', 'BEFORE'): 'BEFORE',
    ('LEQ', 'EQUAL'): 'LEQ',
    ('LEQ', 'LEQ'): 'LEQ',

    ('EQUAL', 'AFTER'): 'AFTER',
    ('EQUAL', 'BEFORE'): 'BEFORE',
    ('EQUAL', 'LEQ'): 'LEQ',
    ('EQUAL', 'GEQ'): 'GEQ',
    ('EQUAL', 'EQUAL'): 'EQUAL',

}


class rel_mult:

    def __init__(self,number_rels,number_events):
        self.number_rels = number_rels
        self.number_events = number_events
        pass



    def mult(self,list1,list2):

        result = None

        for i in range(len(list1)):
            res = list1[i]+"."+list2
            if result is None:
                result = res
            else:
                result = result + "+"+res

        return result


    def sum(self,string1):

        result = None
        total_sum = string1

        list1 = total_sum.split(" + ")
        # print(list1)
        event_coef_dict = {}

        for l in list1:
            if l!="":
                sp = l.split(".")
                coef = sp[0]
                event = sp[1]
                # print(coef)
                # print(event)

                if event_coef_dict.get(event,None) is None:
                    event_coef_dict[event]=[coef]
                    # print(event_coef_dict[event])

                else:
                    # print(event_coef_dict[event])
                    event_coef_dict[event].append(coef)
                    # print(event_coef_dict[event])

                # print("===========")

        for k,v in event_coef_dict.items():
            # print(k)
            # print(event_coef_dict[k])
            # print(len(v))
            # print("===========")
            res = "("+" + ".join(v)+")."+k+"\n"
            if result is None:
                result = res
            else:
                result = result + " + "+res


        return result


    def equation(self):
        inv_map = {v: k for k, v in RELATIONS_DICT.iteritems()}

        lamda = [] #lamda[i][j][k][r1][r2]
        for i in range(self.number_events):
            lamda.append([])
            for j in range(self.number_events):
                lamda[i].append([])
                for k in range(self.number_events):
                    lamda[i][j].append([])
                    for r1 in range(self.number_rels):
                        lamda[i][j][k].append([])
                        for r2 in range(self.number_rels):
                            lamda[i][j][k][r1].append("l_{}_{}_{}_{}_{}".format(r_dict[r1],r_dict[r2],i,j,k))

        # lamda = [] #lamda[i][j][k]
        # for i in range(self.number_events):
        #     lamda.append([])
        #     for j in range(self.number_events):
        #         lamda[i].append([])
        #         for k in range(self.number_events):
        #             lamda[i][j].append("l_{}_{}_{}".format(i,j,k))



        I = [] #I[i][j][r]

        for i in range(self.number_events):
            I.append([])
            for j in range(self.number_events):
                I[i].append([])
                for r in range(self.number_rels):
                    I[i][j].append("I_{}_{}_{}".format(r_dict[r],i,j))


        # I = [] #I[i][j]
        #
        # for i in range(self.number_events):
        #     I.append([])
        #     for j in range(self.number_events):
        #         I[i].append("I_{}_{}".format(i,j))


        # total_sum = None
        # for i in range(self.number_events):
        #     for j in range(self.number_events):
        #         for r1 in range(self.number_rels):
        #             lamda_sum = None
        #             for k in range(self.number_events):
        #                     for r2 in range(self.number_rels):
        #                         if lamda_sum is None:
        #                             lamda_sum = lamda[i][j][k][r1][r2]
        #                         else:
        #                             lamda_sum = lamda_sum + "+" +lamda[i][j][k][r1][r2]
        #
        #             if total_sum is None:
        #                 total_sum = "("+lamda_sum+")."+I[i][j][r1]
        #
        #             else:
        #                 total_sum = total_sum + "+" + "(" + lamda_sum + ")." + I[i][j][r1]
        #
        # total_sum = None
        # for j in range(self.number_events):
        #     for k in range(self.number_events):
        #         for r2 in range(self.number_rels):
        #             lamda_sum = None
        #             for i in range(self.number_events):
        #                     for r1 in range(self.number_rels):
        #                         if lamda_sum is None:
        #                             lamda_sum = lamda[i][j][k][r1][r2]
        #                         else:
        #                             lamda_sum = lamda_sum + "+" +lamda[i][j][k][r1][r2]
        #
        #             if total_sum is None:
        #                 total_sum = "("+lamda_sum+")."+I[j][k][r2]
        #
        #             else:
        #                 total_sum = total_sum + "+" + "(" + lamda_sum + ")." + I[j][k][r2]



        total_sum = None
        for i in range(self.number_events):
            for j in range(self.number_events):
                for k in range(self.number_events):
                    for r1 in range(self.number_rels):
                        for r2 in range(self.number_rels):

                            r3 = RELATIONS_COMPOSITION.get((inv_map[r1].upper(),inv_map[r2].upper()),None)
                            if r3 is not None:
                                # print(i,j,k)
                                if i != j and j != k and i != k:
                                    r3 = RELATIONS_DICT[r3]
                                    sum = lamda[i][j][k][r1][r2]+ "."+I[i][j][r1] + " + " + lamda[i][j][k][r1][r2]+ "."+I[j][k][r2] + " + " + lamda[i][j][k][r1][r2]+ "."+I[i][k][r3]

                                    if total_sum is None:
                                        total_sum = sum
                                    else:
                                        total_sum = total_sum + " + " + sum
        # total_sum = ""
        # for i in range(self.number_events):
        #     for j in range(i+1,self.number_events):
        #             for k in range(j+1,self.number_events):
        #                 # print(lamda[i][j][k],I[i][j])
        #                 total_sum = total_sum + " + " + lamda[i][j][k] + "." + I[i][j]
        #                 total_sum = total_sum + " + " + lamda[i][j][k] + "." + I[j][k]
        #                 total_sum = total_sum + " + " + lamda[i][j][k] + "." + I[i][k]




        # print(total_sum)
        print(self.sum(total_sum))



if __name__ == '__main__':
    rm = rel_mult(5,4)

    rm.equation()
    # currDir = "/home/opandit/Downloads/CA_finals_RTP/SFM"
    #
    # print currDir
    # for filename in os.listdir(currDir):
    #     old_file = os.path.join(currDir,filename)
    #     print(old_file)
    #     new_file = os.path.join(currDir,"RTP_"+filename)
    #     print(new_file)
    #     os.rename(old_file, new_file)








