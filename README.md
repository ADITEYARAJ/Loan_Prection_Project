# Loan_Prection_Project
Check for Defaulter or NotDefaulter
Web app- : https://loan-prediction-123.herokuapp.com/

Understanding the problem statement is the first and foremost step. This would help you give an intuition of what you will face ahead of time. Let us see the problem statement -

A Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.

It is a classification problem where we have to predict whether a loan would be approved or not. In a classification problem, we have to predict discrete values based on a given set of independent variable(s). Classification can be of two types:

Binary Classification : In this classification we have to predict either of the two given classes. For example: classifying the gender as male or female, predicting the result as win or loss, etc. Multiclass Classification : Here we have to classify the data into three or more classes. For example: classifying a movie's genre as comedy, action or romantic, classify fruits as oranges, apples, or pears, etc.

Loan prediction is a very common real-life problem that each retail bank faces atleast once in its lifetime. If done correctly, it can save a lot of man hours at the end of a retail bank.

Dataset- The data has 149999 rows and 26 columns.

Dataset Description-

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



