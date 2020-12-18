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



class BayesClassifier:
    
    def __init__(self):
        global train_labels_list , train_feature_matrix
        self.train_labels_list = train_labels_list
        self.train_feature_matrix = train_feature_matrix
        self.class_MeanVec_CovMat_dict = {}
        unique_labels = set(train_labels_list)
        for class_name in unique_labels:
            tup = self.MeanVec_CovMatrix_of_class_conditional_distibution(class_name)
            self.class_MeanVec_CovMat_dict[class_name] = tup
        
    
    def MeanVec_CovMatrix_of_class_conditional_distibution(self,class_name):
        train_labels_list = self.train_labels_list
        train_feature_matrix = self.train_feature_matrix
        d = (train_feature_matrix.shape)[1]
        n = 0
        mu_vec = numpy.zeros((d,1))
        for i in range(len(train_labels_list)):
            if train_labels_list[i] == class_name:
                n = n+1
                Xi = train_feature_matrix[i:i+1,:]
                Xi = Xi.transpose()
                mu_vec = numpy.add(mu_vec,Xi)
        mu_vec = mu_vec/n
        #print(mu_vec)

        n = 0
        cov_mat = numpy.zeros((d,d))
        for i in range(len(train_labels_list)):
            if train_labels_list[i] == class_name:
                n = n+1
                Xi = train_feature_matrix[i:i+1,:]
                Xi = Xi.transpose()
                temp_vec = numpy.subtract(Xi,mu_vec)
                #print("Xi:" , Xi.shape)
                #print("mu_vec:" , mu_vec.shape)
                #print("temp: ", temp_vec)
                #print("Printed temp")
                #print(numpy.matmul(temp_vec, temp_vec.transpose()))
                cov_mat = numpy.add(cov_mat, numpy.matmul(temp_vec, temp_vec.transpose()))  # summation of (Xi - u)transpose((Xi - u))
                #print("mat:",cov_mat)
        cov_mat = cov_mat/n
        #print("Cov_Mat: " , cov_mat)
        #print("-----------------------------------------------------------------------Done for a class")
        return (mu_vec,cov_mat)
    
    
    def Priori(self,class_label): #Assume equal priors
        return 1/len(set(self.train_labels_list)) # So 1/3 for the given problem
    
    def Discriminant_function_for_given_class(self,class_label , feature_vec):
        feature_vec = numpy.reshape(feature_vec,(-1,1))
        #(MeanVec,CovMat) = self.MeanVec_CovMatrix_of_class_conditional_distibution(class_label)
        (MeanVec,CovMat) =  self.class_MeanVec_CovMat_dict[class_label]
        temp_vec = numpy.subtract(feature_vec,MeanVec)
        G = -1/2 * numpy.matmul( temp_vec.transpose() , numpy.matmul(numpy.linalg.inv(CovMat),temp_vec) )  -  1/2 * math.log(abs(numpy.linalg.det(CovMat))) + math.log(self.Priori(class_label))
        #print("disc=" , G , " class = " , class_label)
        return G
    
    
    
    def Classify_input_feature_vector(self,feature_vec):
        train_labels_list = self.train_labels_list
        unique_labels = set(train_labels_list)
        output_label = "No Label yet..."
        old_G = -10**45
        for label in unique_labels:
            new_G =  self.Discriminant_function_for_given_class(label , feature_vec)
            if old_G < new_G:
                old_G = new_G
                output_label = label
        #print(output_label)
        return output_label
    
    
    def Accuracy(self):  #returns accuracy in percentage
        test_feature_matrix, test_labels_list =  read_data_from_file("test.csv")
        correct_ans = 0;
        for i in range(len(test_labels_list)):
            X = test_feature_matrix[i]
            #X = test_feature_matrix[i,1:3] #---> For testing in case only 2 features are considered...
            test_data_label = test_labels_list[i]
            output_label = self.Classify_input_feature_vector(X)
            if output_label == test_data_label:
                correct_ans = correct_ans + 1
            print("\nSNO)",i, "  feature vector:" , X, "  Output by Bayes Classifier:" , output_label , " Test label: " , test_data_label, "Correctness:",output_label == test_data_label)
        accuracy = (correct_ans/len(test_labels_list))*100
        print("Accuracy of Bayes Classifier(when class conditional distribution is joint Normal) on test set is:", accuracy, "%")
        
    
    def UserInput(self):
        print("\nEnter the feature vector of the object to be classified:")
        X = []
        for i in range((self.train_feature_matrix.shape)[1]):
            print("Enter the feature at place - " ,i, "in the feature vector:")
            xi = float(input())
            X.append(xi)
        output_label = self.Classify_input_feature_vector(numpy.array(X))
        print("Bayes Classifier decides the label for the input feature vector to be:" , output_label)
        



            
print("\n\n----------------------------------------------------------------------------------------------------------------------------------------------------")
#read data from train.csv
print("    Reading data from train.csv...")
train_feature_matrix, train_labels_list =  read_data_from_file("train.csv")
#print(train_feature_matrix.shape)


print("\n\n--------------------------------------------------------------------------------------------------------------------------------------------------------")
print("                                                                             Bayes Classifier                             ")
bayes_classifier =  BayesClassifier() 
bayes_classifier.Accuracy()
bayes_classifier.UserInput()
# (MeanVec,CovMat) = bayes_classifier.MeanVec_CovMatrix_of_class_conditional_distibution("Iris-setosa")
# print(MeanVec)
# print(CovMat)
# print(numpy.linalg.det(CovMat))

# bayes_classifier.Discriminant_function_for_given_class("Iris-versicolor",numpy.array([2.6,2.8,8.6,2.671]))

# output_label = bayes_classifier.Classify_input_feature_vector(numpy.array([6.4,2.8,5.6,2.1]))
# print(output_label)

          