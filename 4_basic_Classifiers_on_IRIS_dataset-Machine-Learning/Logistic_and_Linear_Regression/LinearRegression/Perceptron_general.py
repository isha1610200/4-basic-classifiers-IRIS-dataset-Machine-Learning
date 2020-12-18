#!/usr/bin/env python
# coding: utf-8

# In[1]:

import xlrd
import csv 
import numpy as np
import math
import matplotlib.pyplot as plt
import random


             

# In[2]:


def read_data(file):
    
    feature_vecs = []
    labels = []
    
    with open(file,"r") as csvfile:
        
        csvreader = csv.reader(csvfile)
        i = 0        
        for row in csvreader:
            if(i == 0):
                pass
            else:
            #print(row)
                feature_vecs.append(row[0:len(row)-1])
                labels.append(row[len(row)-1])
            i += 1
            
    return (feature_vecs,labels)
            
 


# In[3]:


class LinearRegressionClassifier:
    
    def __init__(self,train_file,test_file):
        self.theta_vec = None
        self.train_features = None
        self.train_labels = None
        self.test_features = None
        self.test_labels = None
        self.initialize(train_file,test_file)
        self.learn_parameters_stochastic_gradient_descent()
        
        #self.learn_parameters_plain_grad_descent()
        
    def initialize(self,file_train,file_test):
        
        #reading training features and labels
        train_data = read_data(file_train)
        train_features_lst = train_data[0]
        train_labels_lst = train_data[1]

        test_data = read_data(file_test)
        test_features_lst = test_data[0]
        test_labels_lst = test_data[1]
        
        for i in range(len(train_features_lst)): #augmenting feature vectors to account for the bias term
            train_features_lst[i] = [1] + train_features_lst[i]

        for i in range(len(test_features_lst)):
        	test_features_lst[i] = [1] + test_features_lst[i]

        labels_set = set(train_labels_lst)
        if len(labels_set) > 2:
        	print("This classifier is for 2 class problem only!!")
        	exit()

        	
        labels_dict = { }
        count = 0
        for i in labels_set: # labels_set has only 2 items 
        	if(count == 0):
        	    labels_dict[i] = 1
        	    count = 1
        	else:
        	    labels_dict[i] = -1



        for i in range(len(train_labels_lst)):
        	train_labels_lst[i] = labels_dict[train_labels_lst[i]]

        for i in range(len(test_labels_lst)):
        	test_labels_lst[i] = labels_dict[test_labels_lst[i]]

        #print("features:",train_features_lst[0:10])
        #print("labs:",train_labels_lst[0:10])
       

        train_features_mat = np.matrix(train_features_lst)
        self.train_features = train_features_mat.astype(np.float)
        
        self.train_labels = train_labels_lst
        
        
        test_features_mat = np.matrix(test_features_lst)
        self.test_features = test_features_mat.astype(np.float)
        
        self.test_labels = test_labels_lst
        
        
        mean_mat = self.train_features.mean(0)
        std_mat = self.train_features.std(0)
        for j in range(self.train_features.shape[1]):
            for i in range(self.train_features.shape[0]):
                if std_mat[0,j] == 0:
                    pass
                else:
                    self.train_features[i,j] = (self.train_features[i,j] - mean_mat[0,j])/std_mat[0,j]


        for j in range(self.test_features.shape[1]):
            for i in range(self.test_features.shape[0]):
                if std_mat[0,j] == 0:
                    pass
                else:
                    self.test_features[i,j] = (self.test_features[i,j] - mean_mat[0,j])/std_mat[0,j]                     
       

        #initialize theta vec with zeroes
        n = train_features_mat[0,:].shape[1] #length of the augemented feature vector
        self.theta_vec = np.zeros([n,1],dtype = float)
        
        


  
    def classify(self,X): #X is a column vector
        o_val = np.matmul(self.theta_vec.transpose(),X)
        if(o_val > 0):
            return 1
        else:
            return -1


    
    def user_input(self):
        pass
    
    def accuracy_test(self):
        correct = 0
        for i in range(self.test_features.shape[0]):
            output_label = self.classify(self.test_features[i,:].transpose())
            test_label = self.test_labels[i]
            print("Sno: ",i,"  output label:",output_label," test_label:",test_label)
            if test_label == output_label:
                print("CORRECT")
                correct += 1
            else:
                #pass            	
                print("WRONG")
        acc = 100*correct/self.test_features.shape[0]
        return acc
    
    def accuracy_train(self):
        correct = 0
        for i in range(self.train_features.shape[0]):
            output_label = self.classify(self.train_features[i,:].transpose())
            train_label = self.train_labels[i]
            print("Sno: ",i,"  output label:",output_label," test_label:",train_label)
            if train_label == output_label:
                print("CORRECT")
                correct += 1
            else:
                #pass            	
                print("WRONG")
        acc = 100*correct/self.train_features.shape[0]
        return acc
    
                  
    def Sum_of_Squared_Errors_Loss(self):
        cost = 0
        N = self.train_features.shape[0]

        for i in range(N):#for each training example
            Xi = self.train_features[i,:].transpose() #A column vector
            #print("cls op:" , self.classify(Xi))
            #print("act lab:", self.train_labels[i])
            cst = (1/2)*((self.train_labels[i] - self.classify(Xi))**2)
            cost = cost + cst
        return cost        
    
    
    def learn_parameters_stochastic_gradient_descent(self,batchsize = 1, eta = 0.0001,epsilon = 0.00001,max_iter= 8000):
        cost_lst = []
        cost = self.Sum_of_Squared_Errors_Loss()
        print("Starting Loss fn value:",cost)
        cost_lst.append(cost)
        n = self.train_features[0,:].shape[1] #length of the augemented feature vector
        self.theta_vec.fill(0)
        
        bs = batchsize
    
        lst = []
        for i in range(self.train_features.shape[0]):
            lst.append(i)

        count_iter = 0
        
        while(count_iter < max_iter):

            
            rand_sample = random.sample(lst,bs)
            # Now considering a stochastic mini-batch
            #For this mini-batch
              
            Sum = np.zeros([self.train_features.shape[1],1],dtype = float)
                    
            for i in rand_sample:
                Xi = self.train_features[i,:].transpose()
                mult = np.matmul(self.theta_vec.transpose(),Xi)
                mult = mult[0,0]
                Sum = Sum +  -1*(self.train_labels[i] - mult)*Xi
                        
            Grad_cost_fn = Sum 
                    
            self.theta_vec = self.theta_vec + -1*eta*Grad_cost_fn
                    
             
            cost = self.Sum_of_Squared_Errors_Loss()
            cost_lst.append(cost)
            norm = np.linalg.norm(np.matrix(Grad_cost_fn),2)
            count_iter += 1
            print("After ", count_iter ," updation, the cost fn value is ", cost, " and norm of update_vec is ", norm)
            
            
            if(norm < epsilon):
                break
                
            
        plt.plot(cost_lst)
        plt.title("Stochastic mimi batch gradient descent ")
        plt.xlabel("Iterations( 1 iteration is for updation of 1 stochastic mini-batch)")
        plt.ylabel("Cost function")
        plt.show()
        
        
        
    def learn_parameters_plain_grad_descent(self,batchsize = 1, eta = 0.001,epochs_limit = 100):
        cost_lst = []
        cost = self.Sum_of_Squared_Errors_Loss()
        print("Starting Loss fn value:",cost)
        cost_lst.append(cost)
        n = self.train_features[0,:].shape[1] #length of the augemented feature vector
        self.theta_vec.fill(0)  
        
        bs = batchsize
        num_minibatches = math.ceil(self.train_features.shape[0]/bs)
        
        count_epochs = 0
        while(count_epochs < epochs_limit):
            
            for j in range(num_minibatches):
                
                start = j*bs
                stop = (j + 1)*bs
                if(stop > self.train_features.shape[0]/bs):
                    stop = self.train_features.shape[0]
                
                # Now considering a mini-batch i.e. start to stop-1,update theta_vec

                 
                Sum = np.zeros([self.train_features.shape[1],1],dtype = float)
                    
                for i in range(start,stop):
                    Xi = self.train_features[i,:].transpose()
                        #print(Xi.shape)
                        #print(self.theta_vec.shape)
                    mult = np.matmul(self.theta_vec.transpose(),Xi)
                    mult = mult[0,0]
                    Sum = Sum +  -1*(self.train_labels[i] - mult)*Xi
                        
                
                Grad_cost_fn = Sum 
                    
                   
                self.theta_vec = self.theta_vec + -1*eta*Grad_cost_fn    
                
                
                
                cost = self.Sum_of_Squared_Errors_Loss()
                #cost = "later"
                cost_lst.append(cost)
                print("After ", count_epochs*num_minibatches + j + 1 ," updation, the cost fn value is ", cost)
                
            count_epochs += 1
            
        plt.plot(cost_lst)
        plt.xlabel("Iterations( 1 iteration is for updation of 1  mini-batch )")
        plt.ylable("Cost function")
        plt.show()
                
                    
                
                
        


# In[4]:


linear_regression_classifier = LinearRegressionClassifier("train_2.csv","test_2.csv")
print("Accuracy over training set is:",linear_regression_classifier.accuracy_train())
print("Accuarcy over test set is:",linear_regression_classifier.accuracy_test())


# In[ ]:





# In[ ]:




    
    


# In[ ]:




