from flask import Flask
from flask import render_template
from flask import request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection as ms
import sklearn.preprocessing as pre
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from csv import writer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/fraud')
def fraud():
    return render_template('index-temp.html')

@app.route('/send_comp_post',methods=['post'])
def send_comp_post():
    first_name = request.form['first_name']
    #umbrella_limit = request.form['umbrella_limit']
    insured_sex = request.form['insured_sex']
    insured_education_level = request.form['insured_education_level']
    insured_occupation = request.form['insured_occupation']
    insured_hobbies = request.form['insured_hobbies']
    #insured_relationship = request.form['insured_relationship']
    incident_type = request.form['incident_type']
    collision_type = request.form['collision_type']
    incident_severity = request.form['incident_severity']
    authorities_contacted = request.form['authorities_contacted']
    #incident_state = request.form['incident_state']
    number_of_vehicles_involved = int(request.form['number_of_vehicles_involved'])
    property_damage = request.form['property_damage']
    bodily_injuries = int(request.form['bodily_injuries'])
    witnesses = int(request.form['witnesses'])
    police_report_available = request.form['police_report_available']
    injury_claim = request.form['injury_claim']
    property_claim = request.form['property_claim']
    vehicle_claim = int(request.form['vehicle_claim'])
    fraud_reported = 'yes'
    last_name=request.form['last_name']
    policy_no = int(request.form['policy_number'])
    age=int(request.form['age'])
    policy_annual_premium=int(request.form['policy_annual_premium'])
    auto_year=int(request.form['auto_year'])



    #data_set = {'age': [ age ],'policy_annual_premium': [ policy_annual_premium ],'insured_sex':[insured_sex],'insured_education_level': [ insured_education_level ],'insured_occupation': [ insured_occupation ],
     #           'insured_hobbies': [ insured_hobbies ],'incident_type': [ incident_type ], 'collision_type': [ collision_type ],
      #          'incident_severity': [ incident_severity ],'authorities_contacted': [ authorities_contacted ],'number_of_vehicles_involved': [ number_of_vehicles_involved ],
       #         'property_damage': [ property_damage ],'bodily_injuries': [ bodily_injuries ],'witnesses': [ witnesses ],'police_report_available': [ police_report_available ],
        #        'vehicle_claim': [ vehicle_claim ],'auto_year':[auto_year],'fraud_reported':[fraud_reported]
         #   }

    #df1 = pd.DataFrame(data_set, columns=['age' ,'policy_annual_premium','insured_sex', 'insured_education_level',
     #   'insured_hobbies',
      # 'incident_type', 'collision_type', 'incident_severity',
      # 'authorities_contacted',
      # 'number_of_vehicles_involved', 'property_damage', 'bodily_injuries',
      # 'witnesses', 'police_report_available',
       # 'vehicle_claim','auto_year','fraud_reported'])

    data_set = {'age': age,
                'policy_annual_premium': policy_annual_premium,
                'insured_sex': insured_sex,
                'insured_education_level': insured_education_level,
                'insured_occupation': insured_occupation,
                'insured_hobbies': insured_hobbies,
                'incident_type': incident_type,
                'collision_type': collision_type,
                'incident_severity': incident_severity,
                'authorities_contacted': authorities_contacted,
                'number_of_vehicles_involved': number_of_vehicles_involved,
                'property_damage': property_damage,
                'bodily_injuries': bodily_injuries,
                'witnesses': witnesses,
                'police_report_available': police_report_available,
                'injury_claim': injury_claim,
                'property_claim': property_claim,
                'vehicle_claim': vehicle_claim,
                'auto_year': auto_year,
                'fraud_reported': fraud_reported
                }
    df1 = pd.DataFrame(data_set, index=[1])
 #=============================================================================================
    df = pd.read_csv('static/dataset.csv')
    df_copy = df
    df_copy.shape
    #df_temp = pd.DataFrame(df_copy.iloc[0, :]).T
    lb = pre.LabelEncoder()
    Features_selected = ['age', 'policy_annual_premium', 'insured_sex', 'insured_education_level', 'insured_occupation',
                         'insured_hobbies', 'incident_type', 'collision_type', 'incident_severity',
                         'authorities_contacted',
                         'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
                         'police_report_available', 'vehicle_claim', 'auto_year', 'fraud_reported']

    df_ML = df_copy[Features_selected]
    df_copy = df_ML
    lt = df_copy.select_dtypes(include='object').columns.to_list()
    # lt = df_copy.columns.to_list()
    data = pd.DataFrame(df_copy.iloc[2, :])
    data = data.T
    df_copy = pd.concat([df_copy, df1], ignore_index=True)
    for x in lt:
        df_copy[x] = lb.fit_transform(df_copy[x])
    df_ML = df_copy[Features_selected]
    df_ML.shape

    y2 = df_copy.shape

    df_test = pre.minmax_scale(df_ML.values)
    df_Norm = pd.DataFrame(data=df_test, columns=df_ML.columns.to_list())
    x = pd.DataFrame(df_Norm.iloc[1000, :])
    x = x.T
    temp = x.drop('fraud_reported', axis=1)
    df_Norm = df_Norm.drop([1000])
    df_features = df_Norm.drop('fraud_reported', axis=1)
    df_ML = df_ML.drop([1000])
    df_outcome = df_ML['fraud_reported']
    x_train, x_test, y_train, y_test = ms.train_test_split(df_features, df_outcome,
                                                           test_size=0.3,
                                                           random_state=121231234)
    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=7)
    dtc.fit(x_train, y_train)
    y_pred_LR = dtc.predict(temp)
    y_pred_DT = dtc.predict(x_test)

    x2 = accuracy_score(y_test, y_pred_DT).round(4)

    value = y_pred_LR[0]
    if value == 0:
        final = 'Genuine'
    else:
        final = 'Fraud'
    def append_list_as_row(file_name, list_of_elem):
        # Open file in append mode
        with open(file_name, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)

    row_contents = [first_name,last_name,policy_no,final]
    # Append a list as new line to an old csv file
    append_list_as_row('C:/Users/AMAL JOSEPH/OneDrive/Documents/data_temp1.csv', row_contents)
    return render_template('single-post.html' ,data=final ,data2=value, data3=first_name+" "+last_name ,data4=policy_no)
if __name__ == '__main__':
    app.run()
