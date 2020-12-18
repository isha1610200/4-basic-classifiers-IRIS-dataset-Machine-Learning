#!/usr/bin/env python
# coding: utf-8

# In[1]:


# all the necessary imports
import xlrd
import csv

import numpy
import random
import math
import matplotlib
import matplotlib.pyplot as plt 


# In[2]:


class LogisticRegressionClassifier: #For a 2-class classification problem
    
    def GetAugmentedTrainingDataset(self,train_filename):
        # To open Workbook/read xlsx file
        wb = xlrd.open_workbook(train_filename)
        sheet = wb.sheet_by_index(0)

        for i in range(sheet.nrows - 1):
            feature_vec = []
            feature_vec.append(1)
            for j in range(sheet.ncols - 1):
                feature_vec.append(sheet.cell_value(i+1,j))
            self.training_labels_list.append(sheet.cell_value(i+1,sheet.ncols - 1))
            self.augmented_training_feature_matrix.append(feature_vec)
            
        labels_set = set(self.training_labels_list)
        label_count = 0
        for label in labels_set:
            self.labels_dict[label] = label_count
            label_count += 1
            if label_count > 2:
                print("This classifier is meant for 2 class problem only.")
                
        for i in range(len(self.training_labels_list)):
            self.training_labels_list[i] = self.labels_dict[self.training_labels_list[i]]
            
        self.augmented_training_feature_matrix = numpy.array(self.augmented_training_feature_matrix)
        self.augmented_training_feature_matrix = self.augmented_training_feature_matrix.astype('float64')
    
             
            
    
    def __init__(self, train_filename, test_filename):
        
        self.labels_dict = {} # Python dict of class lables:class code
        self.augmented_training_feature_matrix = [] # A numpy matrix of training egs feature vecs
        self.training_labels_list = [] # Python list of training class labels
        
        self.GetAugmentedTrainingDataset(train_filename)
        
        #print(self.labels_dict)
        #print(self.augmented_training_feature_matrix)
        #print(self.training_labels_list)
        self.parameter_vector = None
        self.LearnParameterVector()
        print("Finally the parameter vector is: ",self.parameter_vector)
        self.test_features_list = None
        self.test_labels_list = None
        self.Read_Test_data(test_filename)
        
        
    
    
    
    def LearnParameterVector(self):
        stop_criteria = int(input("Pls enter the stopping criteria: "))
        if stop_criteria == 1:
            self.LearnParaVec_StoppingCriteria1(batchsize = math.ceil(self.augmented_training_feature_matrix.shape[0]/self.augmented_training_feature_matrix.shape[0]))
        else:
            self.LearnParaVec_StoppingCriteria2(batchsize = 1)
            
            
    def sigmoid(self,z): # Activation function is sigmoid in Logistic Regression
        #print("z in sigmoid is:",z)
        #return 1/(1 + math.exp(-z))
        try:
            ans = 1/(1 + math.exp(-z))
        except OverflowError:
            ans = 0.000000000000000000000000000000001
        if(ans == 1.0):
            return 0.999999999999999
        return ans
    
    
    def g(self,X):
        #print(X)
        #print("para vec type:",type(self.parameter_vector), " shape:",self.parameter_vector.shape, " type:",self.parameter_vector.dtype)
        #print("feature_vec_type:",type(X.transpose()) , " shape:" , X.transpose().shape, " type: ", X.transpose().dtype)
        return numpy.dot(self.parameter_vector,X.transpose())
        
            
    
    def Cost(self):
        cost = 0
        for i in range(self.augmented_training_feature_matrix.shape[0]): #Calculate cost 
            X = self.augmented_training_feature_matrix[i]
            y = self.training_labels_list[i]
            h =  self.sigmoid(self.g(X))
            cost = cost + -1*(y*math.log(h) + (1-y)*math.log(1-h))
        return cost
        
    
    def LearnParaVec_StoppingCriteria1(self,eta = 0.02,batchsize = 1,epsilon = 0.01):
        self.parameter_vector = numpy.zeros(self.augmented_training_feature_matrix.shape[1])
        iter = 0
        cost_list = []
        cost_list.append(self.Cost())
        print("batchsize:",batchsize)
        print("Number of training egs:",self.augmented_training_feature_matrix.shape[0])
        print("Epsilon:",epsilon)
        print("Learning rate:",eta)
            
        print("Initial Cost: ",cost_list[0])
        print("------------START----------------\n")
        while(True): 
            
            randomlist = random.sample(range(0, self.augmented_training_feature_matrix.shape[0]), batchsize)
            #print(randomlist)
            
            update_vec = numpy.zeros(self.augmented_training_feature_matrix.shape[1])
           
            
            for i in randomlist:
                X = self.augmented_training_feature_matrix[i]
                update_vec = update_vec + -1*(self.training_labels_list[i] - self.sigmoid(self.g(X)))*X  #Acc to loss function
                
            
            self.parameter_vector =  self.parameter_vector - eta*update_vec #Gradient descent...
            iter += 1
            
            cst = self.Cost()
            cost_list.append(cst)
            print("  Cost after updation:", cst)
            print("  Norm of update_vec: ", numpy.linalg.norm(update_vec,2))
            print("Was iteration:", iter)
            print("---------------------------------------------------------------- ")
            
            if numpy.linalg.norm(update_vec,2) < epsilon:
                break
            
        plt.plot(cost_list)
        plt.xlabel('Mini-batches #   , Mini-batch size =' +  str(batchsize))
        plt.ylabel('Cost(theta)')
        plt.title("Cost vs num_of_iterations(or mini-batches) in Stopping Criteria 1 when learning rate = " + str(eta) + " epsilon = " + str(epsilon))
        plt.show()
    
    
    def LearnParaVec_StoppingCriteria2(self,eta = 0.02, batchsize = 1, epochs_limit = 80):
        self.parameter_vector = numpy.zeros(self.augmented_training_feature_matrix.shape[1])
        epoch_num = 0
        iter = 0
        cost_list = []
        cost_list.append(self.Cost())
        num_mini_batches = math.ceil(self.augmented_training_feature_matrix.shape[0]/batchsize)
        print("batchsize:",batchsize)
        print("Number of training egs:",self.augmented_training_feature_matrix.shape[0])
        print("Num_min_batches:",num_mini_batches)
        print("Epochs limit:",epochs_limit)
        print("Learning rate:",eta)
            
        print("Initial Cost: ",cost_list[0])
        print("------------START----------------\n")
        while(epoch_num < epochs_limit):
            
            for m in range(num_mini_batches):
                # For a mini-batch do this-->
                start = m*batchsize
                end = start + batchsize
                
                if(end > self.augmented_training_feature_matrix.shape[0]):
                    end = self.augmented_training_feature_matrix.shape[0] 
                        
                update_vec = numpy.zeros(self.augmented_training_feature_matrix.shape[1])  
                                
                for i in range(start,end):
                    X = self.augmented_training_feature_matrix[i]
                    update_vec = update_vec + -1*(self.training_labels_list[i] - self.sigmoid(self.g(X)))*X  #Acc to loss function

                self.parameter_vector =  self.parameter_vector - eta*update_vec #Gradient descent...
                
                    
                cst = self.Cost()
                cost_list.append(cst)
                print("  Cost after updation:", cst)
                print("  Norm of update_vec: ", numpy.linalg.norm(update_vec,2))
                iter += 1
                print("Was iteration: ",iter)
                print("---------------------------------------------------------------- ")
                      
            epoch_num += 1
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
            print("Was epoch number: ",epoch_num)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    
        #plt.plot(cost_list)
        #plt.xlabel('Mini-batches #   , Mini-batch size =' +  str(batchsize))
        #plt.ylabel('Cost(theta)')
        #plt.title("Cost vs num_of_iterations(or mini-batches) in Stopping Criteria 2 when learning rate = " + str(eta) + " epochs limit = " + str(epochs_limit))
        #plt.show()    
                
    def ClassifyGivenFeatureVec(self,feature_lst_original):
        feature_lst = feature_lst_original[:]
        feature_lst.insert(0,1)
        feature_vector = numpy.array(feature_lst)
        output = self.sigmoid(self.g(feature_vector))
        class_0 = ""
        class_1 = ""
        for label,code in self.labels_dict.items():
            if code == 0:
                class_0 = label
            elif code == 1:
                class_1 = label
                
        if(output < 0.5):
            return class_0
        elif(output > 0.5):
            return class_1
        else:
            print("Arbitrary decision")
            return class_0 if random.randint(0,1) == 0 else class_1
        
        
    def ClassifyUserInput(self):
        print("Pls enter the feature vector you want to be classified:")
        feature_lst = []
        for i in range(self.augmented_training_feature_matrix.shape[1] - 1):
            feature_val = float(input("Pls enter the "+ str(i) + " feature value:"))
            feature_lst.append(feature_val)
        print("The label outputted by Logistic Regression Classifier is:" , self.ClassifyGivenFeatureVec(feature_lst))
                
    
    
    
    def Read_Test_data(self,test_filename):
        raw_data = open(test_filename,"rt")
        reader = csv.reader(raw_data,delimiter = ",",quoting=csv.QUOTE_NONE)
        test_data = list(reader)
        test_data = test_data[1:]

        test_features_list = []
        test_labels_list = []
        
        for lst in test_data:
            test_feature_vec = lst[0:len(lst)-1]
            test_class_label = lst[len(lst)-1]
            test_feature_vec = [float(j) for j in test_feature_vec]
            test_features_list.append(test_feature_vec)
            test_labels_list.append(test_class_label)
        
        self.test_features_list = test_features_list
        self.test_labels_list = test_labels_list

    
    def Accuracy(self):
        correct = 0
        #print(self.test_features_list)
        for i in range(len(self.test_features_list)):
            X = self.test_features_list[i]
            test_label =  self.test_labels_list[i]
            output_label = self.ClassifyGivenFeatureVec(X)
            if(test_label == output_label):
                correct = correct + 1
                print("Sno:", i, "Test label: ", test_label, "  Output label:", output_label, "  CORRECT")
            else:
                print("Sno:", i, "Test label: ", test_label, "  Output label:", output_label, "  WRONG")
                
        print("Accuracy in percentage of Logistic Regression Classifier is : ", 100*correct/len(self.test_features_list))
        return 100*correct/len(self.test_features_list)
        
   
    
    def Observation1_Variation_accuracy_vs_learning_rate(self):
        bs = 32
        epochs_lim = 80
        eta_list = []
        accuracy_list = []
        eta = 2
        while eta > 0.0000002:
            self.LearnParaVec_StoppingCriteria2(eta = eta , batchsize = bs, epochs_limit = epochs_lim)
            acc = self.Accuracy()
            eta_list.append(eta)
            accuracy_list.append(acc)
            eta = eta/10
        plt.plot(eta_list,accuracy_list)
        plt.xlabel("learning rate")
        plt.ylabel("Accuarcy")
        plt.title("Accuracy vs Leaning rate when batchsize = " +  str(bs) +  " epochs_limit = " + str(epochs_lim))
        plt.show()
    
    def Observation2_Variation_accuracy_vs_batchsize(self):
        eta = 0.02
        epochs_lim = 80
        bs_list = []
        accuracy_list = []
        bs = 1
        while bs <= self.augmented_training_feature_matrix.shape[0]:
            self.LearnParaVec_StoppingCriteria2(eta = eta ,batchsize = bs, epochs_limit = epochs_lim)
            acc = self.Accuracy()
            bs_list.append(bs)
            accuracy_list.append(acc)
            bs = bs + 1
        plt.plot(bs_list,accuracy_list)
        plt.xlabel("Batchsize")
        plt.ylabel("Accuarcy")
        plt.title("Accuracy vs Batchsize when learning rate = " + str(eta) + " epochs_limit = " + str(epochs_lim))
        plt.show()
            
    def Observation3_Variation_accurcay_vs_epochs_lim(self):
        epochs_list = []
        accuracy_list = []
        eta = 0.02
        bs = 32
        epochs_lim = 60
        while epochs_lim <= 100:
            self.LearnParaVec_StoppingCriteria2(eta = eta, batchsize = bs ,epochs_limit = epochs_lim)
            acc = self.Accuracy()
            epochs_list.append(epochs_lim)
            accuracy_list.append(acc)
            epochs_lim = epochs_lim + 1
        plt.plot(epochs_list,accuracy_list)
        plt.xlabel("Epochs")
        plt.ylabel("Accuarcy")
        plt.title("Accuracy vs Number of epochs when learning rate = " + str(eta) + " batchsize = " + str(bs))
        plt.show()
            


# In[3]:


Classifier = LogisticRegressionClassifier("train_2.xlsx", "test_2.csv")


# In[4]:


Classifier.Accuracy()
print("------------------------------------------------------------------------------------------------")
Classifier.ClassifyUserInput()
print("------------------------------------------------------------------------------------------------")
Classifier.Observation1_Variation_accuracy_vs_learning_rate()
print("------------------------------------------------------------------------------------------------")
Classifier.Observation2_Variation_accuracy_vs_batchsize()
print("------------------------------------------------------------------------------------------------")
Classifier.Observation3_Variation_accurcay_vs_epochs_lim()
print("------------------------------------------------------------------------------------------------")




