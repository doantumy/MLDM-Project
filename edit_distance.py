# Ref: Group Project from 1st semester
# https://github.com/ajoseph12/Protein_Alignment_Minimum_Edit_Distance
import sys
import numpy as np
import pandas as pd

class edit_distance:
    
    def __init__(self,fc_1,fc_2,k):
        
        self.fc_1 = ' ' + fc_1  #Source freeman chain code
        self.fc_2 = ' ' + fc_2  #Target freeman chain code
        self.len_1 = len(self.fc_1)
        self.len_2 = len(self.fc_2)
        self.k = k  #Value for the k-strip
        self.df = self.cost_matrix() #Cost Matrix with manually selected values
        self.min = min(self.len_1,self.len_2)
        self.max = max(self.len_1,self.len_2)
        #self.cal_matrix = np.ones(shape = (self.len_2,self.len_1))*float("inf")  #Matrix for calculating the edit distance
        self.cal_matrix = np.ones(shape = (len(self.fc_2),len(self.fc_1)))*float("inf")
        self.distance = None
        

    # Cost Matrix with manually selected values. The selected values is for the Freeman Primitives
                    # Cost of same item is 0
                    # Cost of empty to a value is 1 and vice versa
                    # Cost of other primitives are incremented by 1 uptill 4 and again decremented (Ex: 0 to 1 = 1 and 0 to 7 = 1)
    def cost_matrix(self):
        
        self.cm = np.random.randint(0, 1, size=81).reshape(9, 9)
        self.names = [_ for _ in ' 01234567']
        self.df = pd.DataFrame(self.cm, index=self.names, columns=self.names)
        for i in range(len(self.cm)):  #Cost of insertion and deletion w.r.t empty character
            for j in range(len(self.cm)):
                self.cm[0][j] = 1 
                self.cm[j][0] = 1
        for i in range(len(self.cm)): #Cost for same character/primitive transformation
            for j in range(len(self.cm)):
                self.cm[i][i] = 0

        for i in range(len(self.cm)): #Hand picked cost for other primitive transformation
            for j in range(len(self.cm) - 3):
                if i and j != 0 and i != j and j > i:
                    self.cm[i][j] = self.cm[i][j - 1] + 1
                if i and j != 0 and i != j and j < i:
                    self.cm[i][j] = self.cm[j][i]
        self.cm[1][6] = 3
        self.cm[1][7] = 2
        self.cm[1][8] = 1
        self.cm[2][6] = 4
        self.cm[2][7] = 3
        self.cm[2][8] = 2
        self.cm[3][6] = 3
        self.cm[3][7] = 4
        self.cm[3][8] = 3
        self.cm[4][6] = 2
        self.cm[4][7] = 3
        self.cm[4][8] = 4
        self.cm[5][6] = 1
        self.cm[5][7] = 2
        self.cm[5][8] = 3

        self.cm[6][1] = 3
        self.cm[6][2] = 4
        self.cm[6][3] = 3
        self.cm[6][4] = 2
        self.cm[6][5] = 1
        self.cm[6][7] = 1
        self.cm[6][8] = 2

        self.cm[7][1] = 2
        self.cm[7][2] = 3
        self.cm[7][3] = 4
        self.cm[7][4] = 3
        self.cm[7][5] = 2
        self.cm[7][6] = 1
        self.cm[7][8] = 1

        self.cm[8][1] = 1
        self.cm[8][2] = 2
        self.cm[8][3] = 3
        self.cm[8][4] = 4
        self.cm[8][5] = 3
        self.cm[8][6] = 2
        self.cm[8][7] = 1
    
        #print(df)
        return self.df
    
    def cost(self,i,j): #Addition of cost w.r.t two primitives
        
        #print("label FC1:",i)
        #print("label FC2:",j)
        value = self.df.loc[str(i), str(j)]
        #print("Label value:",value)
        return value
    
    def cal_distance(self): #Calculation of distance using k-strip
        
        if self.len_1 > self.len_2:
            #print("string FC1 is greater")
            
            #print(self.cal_matrix)
        
            for j in range(0,self.len_1):
                #print("j:",j)
                i = (j*self.min)//self.max
                #print("i_f:",i)

                for i in range (max(int(i-self.k),0),min(self.len_2,int(i+self.k+1))):
                    #print("i_s:",i)
                    if j == 0 and i == 0:
                        self.cal_matrix[i,j] = 0
                        #print("i,j=0;;;;\n",self.cal_matrix)
                    elif j == 0:
                        self.cal_matrix[i,j] = i
                        #print("j=0;;;;\n",self.cal_matrix)
                    elif i == 0:
                        self.cal_matrix[i,j] = j
                        #print("i=0;;;;\n",self.cal_matrix)

                    else:
                        if self.fc_2[i] == self.fc_1[j]:
                            #print("FC1:",self.fc_1[j])
                            #print("FC2:",self.fc_2[i])
                            self.cal_matrix[i,j] = self.cal_matrix[i-1,j-1]
                            #print("char equal;;;;\n",self.cal_matrix)
                                  
                        else:
                                #print("FC1:",self.fc_1[j])
                                #print("FC2:",self.fc_2[i])
                                self.cal_matrix[i,j] = min(self.cal_matrix[i,j-1],
                                                          self.cal_matrix[i-1,j],
                                                          self.cal_matrix[i-1,j-1])+self.cost(self.fc_1[j],self.fc_2[i])
                                #print("char not equal;;;;\n",self.cal_matrix)

        else:
            #print("string FC2 is greater")
            
            #print(self.cal_matrix)
            for i in range(0,self.len_2):
                #print("i:",i)
                j = (i*self.min)//self.max
                #print("j_f:",j)
                for j in range (int(max(j-self.k,0)),min(self.len_1,int(j+self.k+1))):
                    #print("j_s:",j)

                    if j == 0 and i == 0:
                        self.cal_matrix[i,j] = 0
                        #print("i,j=0;;;;\n",self.cal_matrix)
                    elif j == 0:
                        self.cal_matrix[i,j] = i
                        #print("j=0;;;;\n",self.cal_matrix)
                    elif i == 0:
                        self.cal_matrix[i,j] = j
                        #print("i=0;;;;\n",self.cal_matrix)

                    else:
                        if self.fc_2[i] == self.fc_1[j]:

                            #print("FC1:",self.fc_1[j])
                            #print("FC2:",self.fc_2[i])
                            self.cal_matrix[i,j] = self.cal_matrix[i-1,j-1]
                            #print("char equal;;;;\n",self.cal_matrix)
                        else:
                            #print("FC1:",self.fc_1[j])
                            #print("FC2:",self.fc_2[i])
                            self.cal_matrix[i,j] = min(self.cal_matrix[i,j-1],
                                                        self.cal_matrix[i-1,j],
                                                        self.cal_matrix[i-1,j-1])+self.cost(self.fc_1[j],self.fc_2[i])
                            #print("char not equal;;;;\n",self.cal_matrix)
        
        
        self.distance = self.cal_matrix[-1, -1].astype(int)
        return self.distance,self.cal_matrix