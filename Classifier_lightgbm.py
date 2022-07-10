import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import myMatrix as mM
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from numpy import genfromtxt
# Light Gradient Boosting Machine
# LightGBM: A Highly Efficient Gradient Boosting Decision Tree
#------------------------------------------------------------
def ComputeKappa(ActualLabel = np.array([1,0,1,1,0,1]), PredictedLabel = np.array([1,0,1,1,1,1])):
    N = ActualLabel.shape[0]
    TP = 0
    for i in range(N):
        if PredictedLabel[i] == 1 and ActualLabel[i] == 1:
            TP += 1
    #print('TP = ', TP)

    FN = 0
    for i in range(N):
        if PredictedLabel[i] == 0 and ActualLabel[i] == 1:
            FN += 1
    #print('FN = ', FN)
    FP = 0
    for i in range(N):
        if PredictedLabel[i] == 1 and ActualLabel[i] == 0:
            FP += 1
    #print('FP = ', FP)

    TN = 0
    for i in range(N):
        if PredictedLabel[i] == 0 and ActualLabel[i] == 0:
            TN += 1
    Kappa_0 = 2*(TP*TN-FN*FP)
    Kappa_1 = (TP+FP)*(FP+TN)+(TP+FN)*(FN+TN)
    Kappa = Kappa_0/Kappa_1
    return Kappa
#------------------------------------------------------------
def RunExperiment(DataLoc = 'D:/FatigueSeverity12000x28_V1.1.csv',
                               featureNum = 28, dataName = 'FatigueSeverity12000x28_V1.1',
                               Test_size_par = 0.3, Par0 = 100, Par1 = 21, Par2 = 6): 
    # Par0 = n_estimators   Par1 = num_leaves   Par2 = max_depth
    # TRN = 20 # Total run number
    mM.CreateFolder("D:/lightgbm_Result/")
    SaveLoc = "D:/lightgbm_Result/" + dataName + "/"
    mM.CreateFolder(SaveLoc)   
    
    dataset = genfromtxt(DataLoc, delimiter=',')

    X = dataset[1:,1:featureNum+1]
    Y = dataset[1:,featureNum+1]

    X = zetascore_table=zscore(X, axis=0)

    ClassLabels = np.unique(Y)
    #print('ClassLabels ', ClassLabels)
    ClassNum = ClassLabels.shape[0]
 
    X = zetascore_table=zscore(X, axis=0)

    # Confusion matrix
    All_CM_test  = np.zeros((ClassNum,ClassNum))
    All_CM_train = np.zeros((ClassNum,ClassNum))         
       
    IdxN_1c = 6
    NumIdx = IdxN_1c*ClassNum*2 # 6 indices for 1 class, 2 phases of train-test

    PerfMatrix_all = np.zeros((NumIdx, 1))   

    # One column: Class0_Train - Class0_Test - Class1_Train - Class1_Test - ...
    
    x_train, x_test, Y_train, Y_test = train_test_split(X, Y, test_size = Test_size_par)                                                        
    # Model Training -------------------------------------------------               
    PredictionModel = lgb.LGBMClassifier(boosting_type='gbdt', 
                                            learning_rate=0.1, n_estimators=100,
                                            num_leaves=21, max_depth=6,
                                            reg_lambda=1, reg_alpha=1)                                            

    PredictionModel.fit(x_train,Y_train, eval_metric='multi_logloss')         
    # Model Prediction -------------------------------------------------
    PredictionModel.fit(x_train, Y_train)
    Y_train_pred = PredictionModel.predict(x_train)
    Y_test_pred = PredictionModel.predict(x_test) 

    Y_train_prob = PredictionModel.predict_proba(x_train)
    Y_test_prob = PredictionModel.predict_proba(x_test)      
        
    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)
    Y_train_pred = Y_train_pred.astype(int)
    Y_test_pred = Y_test_pred.astype(int) 
      
    PerfMatrix = MultiClassModelPerformance(Y_train, Y_train_pred, Y_train_prob,
                                  Y_test, Y_test_pred, Y_test_prob,
                                  1, SaveLoc)    

    mM.WriteCsv(PerfMatrix, SaveLoc + "PerfMatrix" + ".csv")  
    # PerfMatrix for a class c includes the following indices:
    #PefMatrix_c = np.zeros((12,1))
    #    PefMatrix_c[0,0] = CAR_train
    #    PefMatrix_c[1,0] = Precision_train
    #    PefMatrix_c[2,0] = Recall_train
    #    PefMatrix_c[3,0] = f1_train
    #    PefMatrix_c[4,0] = roc_auc_train
    #    PefMatrix_c[5,0] = Kappa_train

    #    PefMatrix_c[6,0] = CAR_test
    #    PefMatrix_c[7,0] = Precision_test
    #    PefMatrix_c[8,0] = Recall_test
    #    PefMatrix_c[9,0] = f1_test
    #    PefMatrix_c[10,0] = roc_auc_test
    #    PefMatrix_c[11,0] = Kappa_test

# --------------------------------------------------------------------------
def MultiClassModelPerformance(Y_train, Y_train_pred, Y_train_prob, Y_test, Y_test_pred, Y_test_prob, r, SaveLoc):
    ClassLabels = np.unique(Y_train)
    #print('ClassLabels ', ClassLabels)
    ClassNumber = ClassLabels.shape[0]
    print('ClassNumber ', ClassNumber)

    Ntr = Y_train.shape[0]
    Nte = Y_test.shape[0]   
    
    PerfMatrix = np.zeros((1,1))
    TotalTestKappa =  0     
    for c in range(ClassNumber):
        print('****************************************')
        print('c = ', c)

        Y_train_c = np.zeros(Ntr)
        Y_test_c = np.zeros(Nte)
        Y_train_pred_c = np.zeros(Ntr)
        Y_test_pred_c = np.zeros(Nte)
       
        Y_train_prob_c = Y_train_prob[:,c]
        Y_test_prob_c = Y_test_prob[:,c]

        for i in range(Ntr):
            if Y_train[i] == c:
                Y_train_c[i] = 1
            if Y_train_pred[i] == c:
                Y_train_pred_c[i] = 1  
                
        for k in range(Nte):
            if Y_test[k] == c:
                Y_test_c[k] = 1    
            if Y_test_pred[k] == c:
                Y_test_pred_c[k] = 1 
        
        #mM.WriteCsv(Y_test_c, 'D:/Y_test_c.csv')
        #mM.WriteCsv(Y_test_pred_c, 'D:/Y_test_pred_c.csv')
        #mM.WriteCsv(Y_test, 'D:/Y_test.csv')

        CAR_train = sum(Y_train_pred_c == Y_train_c)/Ntr
        Precision_train = precision_score(Y_train_c, Y_train_pred_c) 
        Recall_train = recall_score(Y_train_c, Y_train_pred_c) 
        f1_train = f1_score(Y_train_c, Y_train_pred_c)

        fpr_train, tpr_train, thresholds_train = roc_curve(Y_train_c.ravel(), Y_train_prob_c.ravel(),
                                                       pos_label=1)
        roc_auc_train = auc(fpr_train, tpr_train)
        Kappa_train = ComputeKappa(Y_train.ravel(), Y_train_pred.ravel())
        
        CAR_test = sum(Y_test_pred_c == Y_test_c)/Nte
        Precision_test = precision_score(Y_test_c, Y_test_pred_c) 
        Recall_test = recall_score(Y_test_c, Y_test_pred_c) 
        f1_test = f1_score(Y_test_c, Y_test_pred_c)   

        fpr_test, tpr_test, thresholds_test = roc_curve(Y_test_c.ravel(), Y_test_prob_c.ravel(),
                                                       pos_label=1)        

        roc_auc_test = auc(fpr_test, tpr_test)
        Kappa_test = ComputeKappa(Y_test_c.ravel(), Y_test_pred_c.ravel())    
        
        print('CAR_train - class ' + str(c) + ': ', CAR_train)
        print('Precision_train - class ' + str(c) + ': ', Precision_train)
        print('Recall_train - class ' + str(c) + ': ', Recall_train)
        print('f1_train - class ' + str(c) + ': ', f1_train)
        print('roc_auc_train - class ' + str(c) + ': ', roc_auc_train)
        print('Kappa_train - class ' + str(c) + ': ', Kappa_train)

        print('***********')
        print('CAR_test - class ' + str(c) + ': ', CAR_test)
        print('Precision_test - class ' + str(c) + ': ', Precision_test)
        print('Recall_test - class ' + str(c) + ': ', Recall_test)
        print('f1_test - class ' + str(c) + ': ', f1_test)
        print('roc_auc_test - class ' + str(c) + ': ', roc_auc_test)
        print('Kappa_test - class ' + str(c) + ': ', Kappa_test)

        roc_train = np.vstack((fpr_train, tpr_train))
        roc_test = np.vstack((fpr_test, tpr_test))       

        PefMatrix_c = np.zeros((12,1))
        PefMatrix_c[0,0] = CAR_train
        PefMatrix_c[1,0] = Precision_train
        PefMatrix_c[2,0] = Recall_train
        PefMatrix_c[3,0] = f1_train
        PefMatrix_c[4,0] = roc_auc_train
        PefMatrix_c[5,0] = Kappa_train

        PefMatrix_c[6,0] = CAR_test
        PefMatrix_c[7,0] = Precision_test
        PefMatrix_c[8,0] = Recall_test
        PefMatrix_c[9,0] = f1_test
        PefMatrix_c[10,0] = roc_auc_test
        PefMatrix_c[11,0] = Kappa_test

        TotalTestKappa = TotalTestKappa + PefMatrix_c[11,0]

        PerfMatrix = np.vstack((PerfMatrix, PefMatrix_c))
    PerfMatrix = np.delete(PerfMatrix, 0, axis = 0)   

    return PerfMatrix
# --------------------------------------------------------------------------
RunExperiment(DataLoc = 'D:/CrackPatternDataSet12000x28.csv',
                               featureNum = 28, dataName = 'CrackPatternDataSet12000x28',
                               Test_size_par = 0.3, Par0 = 200, Par1 = 41, Par2 = 3)
#--------------------------------------------------------------------------
 