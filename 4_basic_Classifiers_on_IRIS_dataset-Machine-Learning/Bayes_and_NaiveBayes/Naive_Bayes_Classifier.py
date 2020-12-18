import csv
import numpy
import math
import sys

# reading in all the data(features,class-labels) from a given file

def read_data_from_file(file_name):

    raw_data = open(file_name,"rt")
    reader = csv.reader(raw_data,delimiter = ",",quoting=csv.QUOTE_NONE)
    data = list(reader)
    data = data[1:]

    features_list = []
    labels_list = []

    for lst in data:
        lst1 = lst[0:4]
        class_label = lst[4]
        features_list.append(lst1)
        labels_list.append(class_label)

    feature_matrix = numpy.array(features_list)
    feature_matrix = feature_matrix.astype('float64')
    
    #So feature_matrix contains features: x1,x2,X3...... for different examples(test/training data)
    #and labels_list contains corresponding class labels.
    
    return (feature_matrix,labels_list)





class NaiveBayesDiscrete:
    
    
    def __init__(self):
        global train_labels_list , train_feature_matrix
        self.train_labels_list = train_labels_list
        self.train_feature_matrix = train_feature_matrix
        self.classes_priori_dict = {}
        for label in set(self.train_labels_list):
            self.classes_priori_dict[label] = self.Priori(label)
        
        
        
    def Priori(self,class_label): # Estimate from the training set
        count = 0
        for label in self.train_labels_list:
            if label == class_label:
                count = count + 1
        #print("Count in Priori: " , count)
        apriori = count/len(self.train_labels_list)
        #print("Apriori:",apriori)
        return apriori
    
    def ClassConditional(self,feature_vec,class_label,discretization_level): 
        train_labels_list = self.train_labels_list
        train_feature_matrix = self.train_feature_matrix
        class_conditional_prob_individual_features = []
        for i in range(feature_vec.shape[0]): # Since feature_vec is a 1D array numpy...
            count = 0
            n = 0
            for k in range(len(train_labels_list)):
                if train_labels_list[k] == class_label :
                    n = n + 1
                    if(discretization_level == 2):
                        if round(train_feature_matrix[k,i]) == round(feature_vec[i]) : # Instead of discretizing the whole training dataset explicitly, we can do it here at the time of comparison...
                            count = count + 1
                    else: # discretization_level == 1
                        if round(train_feature_matrix[k,i],1) == round(feature_vec[i],1): #Use this, if we don't want to disctetize further down to the level of integers...by rounding off to the nearest integer
                            count = count + 1
            class_conditional_prob_individual_features.append(count/n)
        
        class_conditional_prob = 1
        for prob in class_conditional_prob_individual_features: # Bcoz of conditional independence...
            class_conditional_prob = class_conditional_prob*prob
        
        return class_conditional_prob
    
    
    def Discriminant_function_for_given_class(self,class_label,feature_vec,discretization_level):
        discriminant = self.ClassConditional(feature_vec,class_label,discretization_level) * self.classes_priori_dict[class_label]
        #discriminant = self.ClassConditional(feature_vec,class_label,discretization_level) * self.Priori(class_label)
        #print("class conditional = ",  self.ClassConditional(feature_vec,class_label)  , " disc = ", discriminant , " priori = " , self.Priori(class_label) )
        return discriminant
    
    def Classify_input_feature_vector(self,feature_vec,discretization_level):
        train_labels_list = self.train_labels_list
        unique_labels = set(train_labels_list)
        output_label = "No Label yet..."
        old_G = -10**45
        for label in unique_labels:
            new_G =  self.Discriminant_function_for_given_class(label,feature_vec,discretization_level)
            #print("G: ",new_G," for class:",label)
            if old_G < new_G:
                old_G = new_G
                output_label = label
        return output_label
        
        
    def Accuracy(self,discretization_level = 2):  #returns accuracy in percentage
        test_feature_matrix, test_labels_list =  read_data_from_file("test.csv")
        correct_ans = 0;
        #discretization_level = int(input("Also,Enter the discretization level you would like to have: 1 for rounding off till 1st decimal place, while 2 for rounding off to the nearest integer"))
        for i in range(len(test_labels_list)):
            X = test_feature_matrix[i]
            test_data_label = test_labels_list[i]
            output_label = self.Classify_input_feature_vector(X,discretization_level)
            if output_label == test_data_label:
                correct_ans = correct_ans + 1
            print("\nSNO)",i, "  feature vector:" , X, "  Output by Naive Bayes:" , output_label , " Test label: " , test_data_label, "Correctness:",output_label == test_data_label)
        #print("Correct ans:", correct_ans)
        accuracy = (correct_ans/len(test_labels_list))*100
        print("\nAccuracy of Naive Bayes Classifier with discretization of features on test set is:" , accuracy, "%")
        if(discretization_level == 2):
            print("considering the discretization of features upto closest integer(rounding off...)")
        else:
            print("considering the discretization of features upto 1 decimal place(rounding off...)")    
       
    
    def UserInput(self):
        print("\nEnter the feature vector of the object to be classified:")
        X = []
        for i in range((self.train_feature_matrix.shape)[1]):
            print("Enter the feature at place - " ,i, "in the feature vector:")
            xi = float(input())
            X.append(xi)
        discretization_level = int(input("Also,Enter the discretization level you would like to have: 1 for rounding off to the 1st decimal place, while 2 for rounding off to the nearest integer: "))
        output_label = self.Classify_input_feature_vector(numpy.array(X),discretization_level)
        print("Naive Bayes Classifier(with Discretization of features) decides the label for the input feature vector to be:" , output_label)

        




print("\n\n----------------------------------------------------------------------------------------------------------------------------------------------------")
#read data from train.csv
print("    Reading data from train.csv...")
train_feature_matrix, train_labels_list =  read_data_from_file("train.csv")
#print(train_feature_matrix.shape)


print("\n\n--------------------------------------------------------------------------------------------------------------------------------------------------------")
print("                                                                              Naive Bayes Classifier                                                     ")
naive_bayes = NaiveBayesDiscrete()
#output_label = naive_bayes.Classify_input_feature_vector(numpy.array([1,2,3,4]),2)
#print(output_label)
naive_bayes.Accuracy(2) # Pass 1 or 2 for different levels of discretization.(1 for rounding off to 1st decimal place while 2 for rounding off to nearest integer)
naive_bayes.UserInput()


