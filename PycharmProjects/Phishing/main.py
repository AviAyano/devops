import copy
import csv
import enum
import functools
from enum import Enum
class Size():
    def __init__(self,row = 0, col = 0):
        self.__row = row
        self.__col = col
    def getRow(self):
        return self.__row
    def getCol(self):
        return self.__col
    def setRow(self,new_row):
        self.__row = new_row
    def setCol(self, new_col):
        self.__col = new_col

# class Web(enum.Enum):
#     Phishing = 1
#     Not_Phishing = 0

def split_data():
    minus_one, one = 0, 0
    try:
        with open('naiveBayes.csv') as file: #naiveBayes.csv phishing.csv
            table = csv.reader(file, delimiter=' ')
            for line in table:
                separator_line = line[0].split(',')
                line_without_index_col = separator_line[1:] #[1:] for phishing
                if line_without_index_col[-1] == '-1':
                    minus_one += 1
                elif line_without_index_col[-1] == '1':
                    one += 1
        print("The total dataset size {}.\nThe legit dataset size {} and phishing dataset size {}.".format(minus_one+one,minus_one, one))
        return(minus_one, one)
    except Exception as e:
        print("We have an error with your file, please check it again.\nPlease check it again.\n")

def table_size(rows,cols,prob_table,dataset_name):
    minus_one,one = 0,0
    table = prob_table
    for row in range(rows):
        if table[row][cols-1] == '-1':
            minus_one += 1
        elif table[row][cols-1] == '1':
            one += 1
    line = "The {} dataset contains {} legit samples and {} phishing samples .".format(dataset_name,minus_one,one)
    return(line,minus_one,one)

def openfile(minus_one,one):
    test_data, test_data_a, test_data_b ,prob_table = [],[], [], []
    counter_b,counter_a,rows_number ,col_number,index ,header= 0,0,0,0,0,0
    delete_header = str(input("Do you have header in the table ?  Y/N\n"))
    delete_index_col = str(input("Do you have Index column ?  Y/N\n"))

    if delete_index_col == 'Y':
        index = 1
    else:
        index = 0
    if delete_header == 'Y':
        header = 1
    else:
        header = 0

    try:
        with open('naiveBayes.csv',newline="") as file: # naiveBayes.csv phishing.csv
            table = csv.reader(file, delimiter=' ')

            for lines in table:
                rows_number += 1
                separator_line = lines[0].split(',')
                col_number = len(separator_line)
                line_without_index_col = separator_line[index:]

                if line_without_index_col[-1] == '-1' and counter_a < (minus_one*0.3):
                    test_data_a.append(line_without_index_col)
                    counter_a +=1
                elif line_without_index_col[-1] == '1' and counter_b < (one*0.3):
                    test_data_b.append(line_without_index_col)
                    counter_b += 1
                elif not(separator_line[0][0].isalpha()):
                    prob_table.append(line_without_index_col)      #insert to phishing table
                # line = ''.join(line_without_index_col)    #convert to string without a comma
                # if len(line) > col_number and rows_number != 1:
                #     col_number = len(line) - line.count('-') +1
        test_data = test_data_a + test_data_b
        #print("The test data is :",test_data,'\n',"The prob table is :",prob_table)
        return (rows_number,col_number,prob_table,test_data)
    except Exception as e:
        print("We have an error with your file, please check it again.\nPlease check it again.\n")
    finally:
        prob_table = prob_table [header:] #without the table header
        return(rows_number,col_number,prob_table,test_data)

    #[print(i,end=' ') for i in pishing_table[2] ]
                #print( line,'\n',col_number,'=',len(line) ,'-',line.count(',') ,'-',line.count('-'))
            #print(','.join(line))

def class_prob(rows,cols, prob_table):
    one, minus_one = 0,0
    for row in range(rows):
        cell_value = prob_table[row][cols-1]
        if cell_value == '1':
            one += 1
        elif cell_value == '-1':
            minus_one += 1
        else:
            print("cell value is unexpected :",cell_value)
    prob_class_one = one / rows
    prob_class_minus_one = minus_one / rows
    #prob_class_zero = zero / rows
    #print("The class probability :","\nOne :",prob_class_one,"\nMinus_one :",prob_class_minus_one,"\nzero :",prob_class_zero)
    return (one,minus_one,prob_class_one,prob_class_minus_one)
    #print("Checking the probability rule equals to 1 :",prob_class_one+prob_class_minus_one+prob_class_zero)

def laplacian_correction(class_list,prob_values_given_class):
    corrected = copy.deepcopy(prob_values_given_class)
    laplacian = sum(class_list) + (len(prob_values_given_class))
    for value in range(len(corrected)):
        if corrected[value] in [0.0, 0]:
            corrected[value] = 1 / laplacian
        else:
            corrected[value] = (class_list[value] + 1 )/ laplacian
    return corrected

def item_set(rows, cols,prob_table):
    temp_list,set_table = [],[0] * (cols)
    set_dict = {}
    for col in range(cols):
        for row in range(rows):
             temp_list.append(prob_table[row][col])
        set_table[col] = len(set(temp_list))
        set_dict.update({col : len(set(temp_list))})
        temp_list = []
    max_set = max(set_table)
    #print("The item",max_set)
    return set_dict,max_set

def attribute_prob(prob_dict, rows, cols, prob_table, class_one, class_minus_one):
    set_dict,num = item_set(rows, cols, prob_table)
    for col in range(cols):
        one_class_list, minus_one_class_list ,temp_list0 ,temp_list1 = [0] * num, [0] * num, [0] * num, [0] * num
        for row in range(rows):
            #prob_table[row] = laplacian_correction(cols, prob_table[row])
            cell_value = prob_table[row][col]
            if cell_value == '1':
                if prob_table[row][cols-1] == '1':  #yes - phishing
                    one_class_list[1] += 1
                elif prob_table[row][cols-1] == '-1':  #no - legit
                    minus_one_class_list[1] += 1
            elif cell_value == '-1':
                if prob_table[row][cols-1] == '1':
                    one_class_list[2] += 1
                elif prob_table[row][cols-1] == '-1':
                    minus_one_class_list[2] += 1
            elif cell_value == '0':
                if prob_table[row][cols-1] == '1':
                    one_class_list[0] += 1
                elif prob_table[row][cols-1] == '-1':
                    minus_one_class_list[0] += 1
            else:
                print("cell value is unexpected :", cell_value)
        # one_class_list.index(0) != set_dict.get(col)
        temp_list1=[0]*set_dict.get(col)
        for final_values in range(set_dict.get(col)):
            temp_list1[final_values] = one_class_list[final_values]/class_one
        if 0 in temp_list1 and final_values == (set_dict.get(col)-1):
            temp_list1 = laplacian_correction(one_class_list,temp_list1)

        temp_list0 = [0] * set_dict.get(col)
        for final_value in range(set_dict.get(col)):
            temp_list0[final_value] = minus_one_class_list[final_value] / class_minus_one
        if 0 in temp_list0 and final_value == (set_dict.get(col)-1):
            temp_list0 = laplacian_correction(minus_one_class_list,temp_list0)

        prob_dict.update({col: [temp_list0,temp_list1]})
    return prob_dict,set_dict

def reduce_lists(rows,cols,table,prob_dict,set_dict,prob_class_one,prob_class_minus_one):
    prob_table = table
    prob_x_given_class_one_list ,prob_x_given_class_minus_one_list = [0]*cols,[0]*cols
    prob_x_given_class_one, prob_x_given_class_minus_one,counter ,legit_counter_hit ,phishing_counter_hit= 0,0,0,0,0
    for row in range(rows):
        for col in range(cols): #for phishing
            #if prob_table[row][cols] == '1':  # yes - phishing v[1]
                if prob_table[row][col] == '0':
                    prob_x_given_class_minus_one_list[col] = prob_dict.get(col)[0][0]
                    prob_x_given_class_one_list[col] = prob_dict.get(col)[1][0]
                elif prob_table[row][col] == '1':
                    prob_x_given_class_minus_one_list[col] = prob_dict.get(col)[0][1]
                    prob_x_given_class_one_list[col] = prob_dict.get(col)[1][1]
                elif prob_table[row][col] == '-1':
                    if set_dict.get(col) > 2:
                        prob_x_given_class_minus_one_list[col] = prob_dict.get(col)[0][2]
                        prob_x_given_class_one_list[col] = prob_dict.get(col)[1][2]
                    elif set_dict.get(col) == 2: #change the -1 index to 0 index
                        prob_x_given_class_minus_one_list[col] = prob_dict.get(col)[0][0]
                        prob_x_given_class_one_list[col] = prob_dict.get(col)[1][0]
                    else:
                        print("Error : You have a column attribute with one variable.")

            #elif #prob_table[row][cols] == '-1':  # no - legit
        prob_x_given_class_one = functools.reduce(lambda x, y: x * y, prob_x_given_class_one_list, 1)
        prob_x_given_class_minus_one = functools.reduce(lambda x, y: x * y, prob_x_given_class_minus_one_list, 1)
        #print("\nThe phishing website detector finds website number {} as: ".format(row),end='')
        result = naive_bayes_classifier(prob_x_given_class_one, prob_class_one, prob_x_given_class_minus_one, prob_class_minus_one )
        #print(result,end=" ")
        if "Legit" in result:
            if prob_table[row][cols-1] == '-1':
                legit_counter_hit += 1

        elif "Phishing" in result:
            if prob_table[row][cols-1] == '1':
                phishing_counter_hit += 1


    return (legit_counter_hit,phishing_counter_hit)


def naive_bayes_classifier(prob_x_given_class_one, prob_class_one, prob_x_given_class_minus_one, prob_class_minus_one):
    classifier = [0,0]
    classifier[0] = prob_x_given_class_minus_one * prob_class_minus_one  # minus one
    classifier[1] = prob_x_given_class_one * prob_class_one
    # print('\n',classifier)
    # print("naive_bayes_classifier:", prob_x_given_class_one, prob_class_one, prob_x_given_class_minus_one, prob_class_minus_one)
    result = classifier.index(max(classifier))
    if result == 0:
        return "{}.".format("Legit")
    elif result == 1:
        return "{}.".format("Phishing")
    else:
        return "{}.".format("Unknown")

def final_porb(sum_of_minus_one, sum_of_one, legit_counter_hit, phishing_counter_hit):
    print_legit,print_phishing = " "," "
    if sum_of_minus_one >= legit_counter_hit:
        print_legit = "\nThe dataset legit web accuracy percentage : {:.2f}".format(legit_counter_hit / sum_of_minus_one)
    else:
        print_legit = "\n Error : sum_of_minus_one < legit_counter_hit - Check the data ! "

    if sum_of_one >= phishing_counter_hit:
        print_phishing = "\nThe dataset phishing web accuracy percentage : {:.2f}".format(phishing_counter_hit / sum_of_one)
    else:
        print_phishing = "\n Error : sum_of_one < phishing_counter_hit - Check the data ! "
    return print_legit,print_phishing
if __name__ == '__main__':

    prob_dict ,test_dict = {}, {}
    minus_one, one = split_data()
    row, col, train_data, test_data = openfile(minus_one, one)
    #[print(i, sep='\n') for i in train_data[:5]]


    train = Size(len(train_data), len(train_data[1]) ) ##-1

    # [print(i, sep='\n') for i in train_data]
    # print("##############")
    # [print(i, sep='\n') for i in test_data]
    print( "The training dataset is: {} X {} .".format(train.getRow(), train.getCol()))
    print_size_train, sum_of_minus_one_trian,sum_of_one_trian = table_size(train.getRow(), train.getCol(), train_data, "train")
    print(print_size_train)
    class_one ,class_minus_one ,prob_class_one ,prob_class_minus_one = class_prob(train.getRow(), train.getCol(), train_data)
    prob_dict,set_dict = attribute_prob(prob_dict, train.getRow(), train.getCol(), train_data, class_one, class_minus_one)
    legit_counter_hit,phishing_counter_hit = reduce_lists(train.getRow(),train.getCol(),train_data,prob_dict,set_dict,prob_class_one,prob_class_minus_one)
    print_legit_trian,print_phishing_trian = final_porb(sum_of_minus_one_trian,sum_of_one_trian,legit_counter_hit,phishing_counter_hit)
    print("\n\n***************** The training dataset results *****************")
    print(print_legit_trian, print_phishing_trian, sep='\n')


    test = Size(len(test_data), len(test_data[1]))
    print("\n\n***************** The test dataset results *****************")
    print("The test dataset is: {} X {} .".format(test.getRow(), test.getCol()))


    print_size_test, sum_of_minus_one_test, sum_of_one_test = table_size(test.getRow(), test.getCol(), test_data,"test")
    print(print_size_test)
    class_one, class_minus_one, prob_class_one, prob_class_minus_one = class_prob(test.getRow(), test.getCol(),test_data)
    test_dict = attribute_prob(test_dict, test.getRow(), test.getCol(), test_data, class_one, class_minus_one)
    legit_counter, phishing_counter = reduce_lists(test.getRow(),test.getCol(),test_data,prob_dict,set_dict,prob_class_one,prob_class_minus_one)
    print_legit_test, print_phishing_test = final_porb(sum_of_minus_one_test, sum_of_one_test ,legit_counter, phishing_counter)
    print(print_legit_test, print_phishing_test, sep='\n')
    #print("\n\n***************** The training dataset results *****************")
    #print(print_legit_trian, print_phishing_trian, sep='\n')
