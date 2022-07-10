import math
import random as rn
import matplotlib.pyplot as plt
import statistics as stat
import numpy as np
import pandas as pd
from numpy import asarray
from numpy import save
from numpy import load
from scipy.stats import zscore
from sklearn import metrics
import os
# Load a library from a specific folder
##import importlib.util
##mMspec = importlib.util.spec_from_file_location("myMatrix",
##                                                "D:/PythonFiles/[0] myLib/myMatrix.py")
##mM = importlib.util.module_from_spec(mMspec)
##mMspec.loader.exec_module(mM)
# --------------------------------------------------
def CreateFolder(path):    
    # path = "D:/SampledImage/"
    if os.path.exists(path) == False:
        os.mkdir(path)
# --------------------------------------------------
def ComputeClassifierPer_3Class(Y_test, Y_test_pred):
    RP_Test = metrics.classification_report(Y_test, Y_test_pred, digits=3,
                                            output_dict = True)   
    # Class 0 Test
##    print('--------------------------------')
    Macro_C0_Test  = RP_Test.get('0')
    Macro_C0_Precision_Test  = Macro_C0_Test .get('precision')
    Macro_C0_Recall_Test  = Macro_C0_Test .get('recall')
    Macro_C0_f1_score_Test  = Macro_C0_Test.get('f1-score')
##    print('Macro_C0_Precision_Test', Macro_C0_Precision_Test)
##    print('Macro_C0_Recall_Test', Macro_C0_Recall_Test)
##    print('Macro_C0_f1_score_Test', Macro_C0_f1_score_Test)
    # Class 1 Test
##    print('--------------------------------')
    Macro_C1_Test  = RP_Test.get('1')
    Macro_C1_Precision_Test  = Macro_C1_Test .get('precision')
    Macro_C1_Recall_Test  = Macro_C1_Test .get('recall')
    Macro_C1_f1_score_Test  = Macro_C1_Test.get('f1-score')
##    print('Macro_C1_Precision_Test', Macro_C1_Precision_Test)
##    print('Macro_C1_Recall_Test', Macro_C1_Recall_Test)
##    print('Macro_C1_f1_score_Test', Macro_C1_f1_score_Test)
    # Class 2 Test
##    print('--------------------------------')
    Macro_C2_Test  = RP_Test.get('2')
    Macro_C2_Precision_Test  = Macro_C2_Test .get('precision')
    Macro_C2_Recall_Test  = Macro_C2_Test .get('recall')
    Macro_C2_f1_score_Test  = Macro_C2_Test.get('f1-score')
##    print('Macro_C2_Precision_Test', Macro_C2_Precision_Test)
##    print('Macro_C2_Recall_Test', Macro_C2_Recall_Test)
##    print('Macro_C2_f1_score_Test', Macro_C2_f1_score_Test)
##    print('--------------------------------')
    # Macro Average Test
    Macro_Average_Test  = RP_Test.get('macro avg')
    Macro_Average_Precision_Test  = Macro_Average_Test .get('precision')
    Macro_Average_Recall_Test  = Macro_Average_Test .get('recall')
    Macro_Average_f1_score_Test  = Macro_Average_Test.get('f1-score')
    CAR_test = sum(Y_test_pred == Y_test)/len(Y_test_pred)
##    print('Macro_Average_Precision_Test', Macro_Average_Precision_Test)
##    print('Macro_Average_Recall_Test', Macro_Average_Recall_Test)
##    print('Macro_Average_f1_score_Test', Macro_Average_f1_score_Test)
##    print('CAR_test', CAR_test_val)

    outPut = np.zeros(13)
    outPut[0] = Macro_C0_Precision_Test
    outPut[1] = Macro_C0_Recall_Test
    outPut[2] = Macro_C0_f1_score_Test
    outPut[3] = Macro_C1_Precision_Test
    outPut[4] = Macro_C1_Recall_Test
    outPut[5] = Macro_C1_f1_score_Test
    outPut[6] = Macro_C2_Precision_Test
    outPut[7] = Macro_C2_Recall_Test
    outPut[8] = Macro_C2_f1_score_Test
    outPut[9] = Macro_Average_Precision_Test
    outPut[10] = Macro_Average_Recall_Test
    outPut[11] = Macro_Average_f1_score_Test
    outPut[12] = CAR_test
    return outPut

# --------------------------------------------------   
def CreateFiveFolds(DataLoc ,FeaNum):
    # DataLoc = 'SFRCBeams_Shear_Hooked275x12.csv'
    X, Y, meanX, stdX, meanY, stdY  = \
    LoadRegCsvData(DataLoc, FeaNum)
    # SoilLiquefaction226x6 SFRCBeams_Shear_Hooked275x12
    N = len(Y)
    # print('N = ', N)

    Sp = np.linspace(0, N, 6)
    Sp = Sp.astype(int)
    # print('Sp = ', Sp)

    idxSet0 = Sp[0:2]
    idxSet1 = Sp[1:3]
    idxSet2 = Sp[2:4]
    idxSet3 = Sp[3:5]
    idxSet4 = Sp[4:6]

    ##print('idxSet0 = ', idxSet0)
    ##print('idxSet1 = ', idxSet1)
    ##print('idxSet2 = ', idxSet2)
    ##print('idxSet3 = ', idxSet3)
    ##print('idxSet4 = ', idxSet4)

    Xtest0 = X[idxSet0[0]:idxSet0[1], :]
    Xtrain0 = np.delete(X, slice(idxSet0[0], idxSet0[1]), 0)
    Xtest1 = X[idxSet1[0]:idxSet1[1], :]
    Xtrain1 = np.delete(X, slice(idxSet1[0], idxSet1[1]), 0)
    Xtest2 = X[idxSet2[0]:idxSet2[1], :]
    Xtrain2 = np.delete(X, slice(idxSet2[0], idxSet2[1]), 0)
    Xtest3 = X[idxSet3[0]:idxSet3[1], :]
    Xtrain3 = np.delete(X, slice(idxSet3[0], idxSet3[1]), 0)
    Xtest4 = X[idxSet4[0]:idxSet4[1], :]
    Xtrain4 = np.delete(X, slice(idxSet4[0], idxSet4[1]), 0)

    Ytest0 = Y[idxSet0[0]:idxSet0[1]]
    Ytrain0 = np.delete(Y, slice(idxSet0[0], idxSet0[1]), 0)
    Ytest1 = Y[idxSet1[0]:idxSet1[1]]
    Ytrain1 = np.delete(Y, slice(idxSet1[0], idxSet1[1]), 0)
    Ytest2 = Y[idxSet2[0]:idxSet2[1]]
    Ytrain2 = np.delete(Y, slice(idxSet2[0], idxSet2[1]), 0)
    Ytest3 = Y[idxSet3[0]:idxSet3[1]]
    Ytrain3 = np.delete(Y, slice(idxSet3[0], idxSet3[1]), 0)
    Ytest4 = Y[idxSet4[0]:idxSet4[1]]
    Ytrain4 = np.delete(Y, slice(idxSet4[0], idxSet4[1]), 0)

    SaveDataAsNpy(Xtrain0, 'Xtrain0.npy')
    SaveDataAsNpy(Xtrain1, 'Xtrain1.npy')
    SaveDataAsNpy(Xtrain2, 'Xtrain2.npy')
    SaveDataAsNpy(Xtrain3, 'Xtrain3.npy')
    SaveDataAsNpy(Xtrain4, 'Xtrain4.npy')

    SaveDataAsNpy(Ytrain0, 'Ytrain0.npy')
    SaveDataAsNpy(Ytrain1, 'Ytrain1.npy')
    SaveDataAsNpy(Ytrain2, 'Ytrain2.npy')
    SaveDataAsNpy(Ytrain3, 'Ytrain3.npy')
    SaveDataAsNpy(Ytrain4, 'Ytrain4.npy')

    SaveDataAsNpy(Xtest0, 'Xtest0.npy')
    SaveDataAsNpy(Xtest1, 'Xtest1.npy')
    SaveDataAsNpy(Xtest2, 'Xtest2.npy')
    SaveDataAsNpy(Xtest3, 'Xtest3.npy')
    SaveDataAsNpy(Xtest4, 'Xtest4.npy')

    SaveDataAsNpy(Ytest0, 'Ytest0.npy')
    SaveDataAsNpy(Ytest1, 'Ytest1.npy')
    SaveDataAsNpy(Ytest2, 'Ytest2.npy')
    SaveDataAsNpy(Ytest3, 'Ytest3.npy')
    SaveDataAsNpy(Ytest4, 'Ytest4.npy')
# --------------------------------------------------

def LoadRegCsvData(Loc, featureNum):
    #Loc = 'SFRCBeams_Shear_Hooked275x12.csv'
    dataset	= pd.read_csv(Loc)
    X	= dataset.iloc[:,0:featureNum].values
    Y	= dataset.iloc[:,featureNum].values
    meanY = np.mean(Y, axis = 0)
    stdY = np.std(Y, axis = 0)
    meanX = np.mean(X, axis = 0)
    stdX = np.std(X, axis = 0)
    X = zetascore_table=zscore(X, axis=0)
    #X_df = pd.DataFrame(X)
    Y = zetascore_table=zscore(Y, axis=0)
    #Y_df = pd.DataFrame(Y)
    return X, Y, meanX, stdX, meanY, stdY 

# --------------------------------------------------
def SaveDataAsNpy(data, SaveLoc):
    # define data
    # SaveLoc = 'data.npy'
    # data = asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    # save to npy file
    save(SaveLoc, data)
# --------------------------------------------------
def LoadNpyData(DataLoc):
    # load numpy array from npy file      
    data = load(DataLoc)
    return data

# --------------------------------------------------
def BinaryColumnSelection(X, Pos_Bin):
##    X = mM.CreateRandomMatrix(5, 4, np.zeros(4), np.ones(4)*10)
##    print('X = \n', X)
##
##    Pos_Bin = [0,1,0,1]
##    print('Pos_Bin = ', Pos_Bin)
##
##    Y = BinaryFeatureSelection(X, Pos_Bin)
##    print('Y = \n', Y)

    D = X.shape[1]

    Pos_Idx =np.arange(D)
    #print('Pos_Idx = ', Pos_Idx)

    Size = np.sum(Pos_Bin)
    #print('Size = ', Size)

    SelPos = np.zeros(Size)
    k = 0
    for i in range(D):
        if Pos_Bin[i] == 1:
            SelPos[k] = int(Pos_Idx[i])
            k = k + 1
    SelPos_int = SelPos.astype(int)      
    # print('SelPos_int = \n', SelPos_int)        

    Y = X[:, SelPos_int]
    # print('Y = \n', Y)
    return Y
# --------------------------------------------------

def CreateRandomMatrix(M = 5, N = 2, LB = [0, 0], UB = [10, 10]):
    X = np.zeros((M,N))
    for i in range(M):
        for k in range(N):
            X[i,k] = rn.uniform(LB[k], UB[k])
    return X
# --------------------------------------------------
def CreateGaussRandMatrix(M = 3, N = 2, mean = 0, std = 1):
    X = np.zeros((M,N))
   
    for i in range(M):
        for j in range(N):
            X[i,j] = rn.gauss(mean, std)
    return X
# --------------------------------------------------
def PrintMatrix(X, Mname):
    print(Mname + ' = \n', X)
# --------------------------------------------------
def TransposeMatrix(X):
    Y = X.T.copy()
    return Y
# --------------------------------------------------
def WriteCsv(Matrix, Loc):
    Matrix_df = pd.DataFrame(Matrix)
    Matrix_df.to_csv(Loc)
# --------------------------------------------------
def ReadCsv(Loc):
    X = pd.read_csv(Loc).values
    # The first column (0) is data index so we start to read data from column 1
    Y = X[1:, 1:]
    return Y
# --------------------------------------------------
    

    

