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
![image](https://user-images.githubusercontent.com/58631474/97907483-5d5cc880-1d6b-11eb-8a2b-d8ee68a53330.png)
## EDA

Disbursement Gross : As per the below graph, more cases have less amount.
        As the disbursement gross amount increases, chances of defaulting decreases.
        
![image](https://user-images.githubusercontent.com/58631474/97907455-546bf700-1d6b-11eb-83f0-62f53aa55c6b.png)     

NewExist: Existing businesses have a marginally more chance to default than new businesses.

![image](https://user-images.githubusercontent.com/58631474/97907570-7a919700-1d6b-11eb-9661-523a02e647dd.png)

Franchisecode:  5260 businesses have franchises and defaulting chances are less for businesses with franchises.

![image](https://user-images.githubusercontent.com/58631474/97907587-81b8a500-1d6b-11eb-8b7e-0b14b89e5bef.png)

Retainedjob:  If no jobs retained defaulting is very less, then the chances of defaulting comes down as the jobs increases.

![image](https://user-images.githubusercontent.com/58631474/97907607-88471c80-1d6b-11eb-97fe-7e87bb1b14ca.png)

UrbanRural: Urban business more likely to default than rural businesses

![image](https://user-images.githubusercontent.com/58631474/97907625-9006c100-1d6b-11eb-8c58-655fcd42671b.png)

Lowdocs: If covered under LowDoc, then very unlikely to default.

![image](https://user-images.githubusercontent.com/58631474/97907646-95fca200-1d6b-11eb-8b55-3b72221215a0.png)

Term: Loans for 0-5 and 30-40 month term has more chance to default, and 5-30 month term less chance of defaulting.

![image](https://user-images.githubusercontent.com/58631474/97907655-9c8b1980-1d6b-11eb-9587-4ba82abf0152.png)

Number of Emp: As the number of employees in the business increase, chances of defaulting decreases.

![image](https://user-images.githubusercontent.com/58631474/97907676-a3199100-1d6b-11eb-9a78-3da1c66d1ff4.png)


Created job: Chances of defaulting is least when jobs created is between 10 and 400, highest when greater than 400.


        
       

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



