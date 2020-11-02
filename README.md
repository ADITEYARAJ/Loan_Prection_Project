# Loan_Prection_Project

## Check for Defaulter or NotDefaulter

### Web app- : https://loan-prediction-123.herokuapp.com/

Understanding the problem statement is the first and foremost step. This would help you give an intuition of what you will face ahead of time. Let us see the problem statement -

1. A Bank accepts deposits from customers and from the corpus thus available, it lends to Borrowers who want to carry out certain Business activities for Growth and Profit. It is often seen that due to some reasons like failure of Business, the company making losses or the company becoming delinquent/bankrupt the loans are either not Paid in full or are Charged-off or are written off. The Bank is thus faced with the problem of identifying those Borrowers who can pay up in full and not lending to borrowers who are likely to default.
2. At the time of giving loan to a new customer or a company, the bank needs to know the chance of getting risk on loan repayment, based on some relevant information.
Predicting the chance of risk is more relevant before giving the loan. So, it reduces the risk of profit & loss of the bank. 
3. When the prediction is showing high risk, the bank have to collect more information about the background of that particular company. Then bank will decide to give the loan or not.


# Dataset- 
The data has 149999 rows and 26 columns.

Dataset Description-

Column names-:
City                      
State                     
Zip                       
Bank                    
BankState               
CCSC                      
ApprovalDate              
ApprovalFY                
Term                      
NoEmp                     
NewExist                  
CreateJob                 
RetainedJob               
FranchiseCode             
UrbanRural                
RevLineCr                
LowDoc                    
ChgOffDate           
DisbursementDate        
DisbursementGross         
BalanceGross              
MIS_Status              
ChgOffPrinGr              
GrAppv                    
SBA_Appv                  



## Project Architecture


![image](https://user-images.githubusercontent.com/58631474/97906721-23d78d80-1d6a-11eb-96d4-274d61cdbfcd.png)

## DATA Cleaning                                

NULL VALUES

![image](https://user-images.githubusercontent.com/58631474/97907120-c98afc80-1d6a-11eb-8a11-9ac1074e4e03.png)

## EDA

Disbursement Gross : As per the below graph, more cases have less amount.
        As the disbursement gross amount increases, chances of defaulting decreases.
        
        
       

Model-:


1.XGBOOST 
 
       precision    recall  f1-score   support

           0       0.87      1.00      0.93      6544
           1       1.00      0.94      0.97     17509

    accuracy                           0.96     24053
   macro avg       0.94      0.97      0.95     24053
weighted avg       0.96      0.96      0.96     24053


2.KNN

          precision    recall  f1-score   support

           0       0.97      1.00      0.98      6544
           1       1.00      0.99      0.99     17509

    accuracy                           0.99     24053
   macro avg       0.98      0.99      0.99     24053
weighted avg       0.99      0.99      0.99     24053


3.RANDOM FOREST

        precision    recall  f1-score   support

           0       0.97      1.00      0.98      6544
           1       1.00      0.99      0.99     17509

    accuracy                           0.99     24053
   macro avg       0.98      0.99      0.99     24053
weighted avg       0.99      0.99      0.99     24053



