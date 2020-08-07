import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 


# Utils
#import os
#import joblib


feature_names_best =['CCSC', 'Term', 'NoEmp', 'NewExist', 'CreateJob', 'RetainedJob',
       'FranchiseCode', 'RevLineCr', 'LowDoc', 'BalanceGross', 'ChgOffPrinGr',
       'SBA_Appv']

# Load ML Models
#def load_model(model_file):
#	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
#	return loaded_model
def get_key(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return key
html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Disease Mortality Prediction </h1>
		<h5 style="color:white;text-align:center;">Hepatitis B </h5>
		</div>
		"""

# Avatar Image using a url
avatar1 ="https://www.w3schools.com/howto/img_avatar1.png"
avatar2 ="https://www.w3schools.com/howto/img_avatar2.png"

result_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

result_temp2 ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/{}" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

prescriptive_message_temp ="""
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Avoid Alchol</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""
def main():
    st.title("Bank Authenticator")
    html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">LOAN DEFAULT PREDICTION </h1>
		<h5 style="color:white;text-align:center;"> </h5>
		</div>
		"""
    result_temp ="""
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">***Algorithm***</h4>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">Result </p>
	</div>
    """
   # Avatar Image using a url
    avatar1 ="https://www.w3schools.com/howto/img_avatar1.png"
    avatar2 ="https://www.w3schools.com/howto/img_avatar2.png"
   
    st.markdown(html_temp,unsafe_allow_html=True)

    st.subheader("Predictive Analytics")
    CCSC=st.sidebar.text_input("CCSC","451120")

    Term=st.sidebar.text_input("Term","480,107,0")
    NoEmp=st.sidebar.text_input("NoEmp","9999,0,9999")
    NewExist=st.sidebar.radio("NewExist","10")
    CreateJob=st.sidebar.text_input("CreateJob","3000,0,3000")
    RetainedJob=st.sidebar.text_input(" RetainedJo","ret")
    FranchiseCode=st.sidebar.text_input("FranchiseCode","91999,91999,0")
    RevLineCr=st.sidebar.radio("RevLineCr","01")
    LowDoc=st.sidebar.selectbox("LowDoc","01")
 
    BalanceGross=st.sidebar.text_input("BalanceGross","827875.0,827875.0,0.0")
    ChgOffPrinGr=st.sidebar.slider("ChgOffPrinGr",1999999.0,1999999.0,0.0)
    SBA_Appv=st.sidebar.slider("SBA_Appv",1500000.0,1500000.0,500.00)
    feature_list =[CCSC,Term,NoEmp,NewExist,CreateJob,RetainedJob,FranchiseCode,RevLineCr,LowDoc,BalanceGross,ChgOffPrinGr,SBA_Appv]
    st.write(len(feature_list))
    st.write(feature_list)
    pretty_result={
                'CCSC':CCSC,
                'Term':Term,
                'NoEmp':NoEmp,
                'NewExist':NewExist,
                'CreateJob':CreateJob,
                'RetainedJob':RetainedJob,
                'FranchiseCode':FranchiseCode,
                'RevLineCr':RevLineCr,
                'LowDoc':LowDoc,
                'BalanceGross':BalanceGross,
                'ChgOffPrinGr':ChgOffPrinGr,
                'SBA_Appv':SBA_Appv
               }
    st.json(pretty_result)
    single_sample = np.array(feature_list).reshape(1,-1)
    model_choice = st.selectbox("Select Model",["KNN","DecisionTree","RandomForest"])
    if st.button("Predict"):
        if model_choice == "KNN":
            loaded_model = joblib.load("knn_model.pkl")
            prediction = loaded_model.predict(single_sample)
            pred_prob = loaded_model.predict_proba(single_sample)
        elif model_choice == "DecisionTree":
            loaded_model = joblib.load("decision_tree_model.pkl")
            prediction = loaded_model.predict(single_sample)
            pred_prob = loaded_model.predict_proba(single_sample)
        else :
            loaded_model = joblib.load("rf_model.pkl")
            prediction = loaded_model.predict(single_sample)
            pred_prob = loaded_model.predict_proba(single_sample)
        st.markdown(result_temp,unsafe_allow_html=True)
        st.write(prediction)
        prediction_label = {"Default":0,"Not Default":1}
        final_result = get_key(prediction,prediction_label)
        if prediction == 1:
            st.success("Not Default")
            pred_probability_score = {"Default":pred_prob[0][0]*100,"Not Default":pred_prob[0][1]*100}
            st.subheader("Prediction Probability Score using {}".format(model_choice))
            st.json(pred_probability_score)
        else:
            st.success("Default")
            pred_probability_score = {"Default":pred_prob[0][0]*100,"Not Default":pred_prob[0][1]*100}
            st.subheader("Prediction Probability Score using {}".format(model_choice))
            st.json(pred_probability_score)
            
if __name__ == '__main__':
	main()
            
        
    
    
    
    
    
    
    
