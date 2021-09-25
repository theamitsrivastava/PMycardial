from flask import Flask, render_template, request
import numpy as np
import joblib
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict/', methods=['GET','POST'])
def predict():
    
    if request.method == "POST":
        #get form data
        AGE = request.form.get('AGE')
        STENOK_AN = request.form.get('STENOK_AN')
        FK_STENOK = request.form.get('FK_STENOK')
        IBS_POST = request.form.get('IBS_POST')
        ZSN_A = request.form.get('ZSN_A')
        nr_04 = request.form.get('nr_04')
        S_AD_KBRIG = request.form.get('S_AD_KBRIG')
        D_AD_KBRIG = request.form.get('D_AD_KBRIG')
        S_AD_ORIT = request.form.get('S_AD_ORIT')
        D_AD_ORIT = request.form.get('D_AD_ORIT')
        K_SH_POST = request.form.get('K_SH_POST')
        ant_im = request.form.get('ant_im')
        lat_im = request.form.get('lat_im')
        ritm_ecg_p_07 = request.form.get('ritm_ecg_p_07')
        n_r_ecg_p_04 = request.form.get('n_r_ecg_p_04')
        n_p_ecg_p_10 = request.form.get('n_p_ecg_p_10')
        n_p_ecg_p_12 = request.form.get('n_p_ecg_p_12')
        K_BLOOD = request.form.get('K_BLOOD')
        NA_BLOOD = request.form.get('NA_BLOOD')
        ALT_BLOOD = request.form.get('ALT_BLOOD')
        AST_BLOOD = request.form.get('AST_BLOOD')
        L_BLOOD = request.form.get('L_BLOOD')
        ROE = request.form.get('ROE')
        TIME_B_S = request.form.get('TIME_B_S')
        R_AB_1_n = request.form.get('R_AB_1_n')
        R_AB_3_n = request.form.get('R_AB_3_n')
        NA_KB = request.form.get('NA_KB')
        NOT_NA_KB = request.form.get('NOT_NA_KB')
        NITR_S = request.form.get('NITR_S')
        NA_R_1_n = request.form.get('NA_R_1_n')
        GEPAR_S_n = request.form.get('GEPAR_S_n')
        RAZRIV = request.form.get('RAZRIV')
        ZSN = request.form.get('ZSN')
        REC_IM = request.form.get('REC_IM')
        DRESSLER = request.form.get('DRESSLER')
     
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(AGE,STENOK_AN,FK_STENOK,IBS_POST,ZSN_A,nr_04,S_AD_KBRIG,D_AD_KBRIG,S_AD_ORIT,D_AD_ORIT,K_SH_POST,ant_im,lat_im,ritm_ecg_p_07,n_r_ecg_p_04,n_p_ecg_p_10,
n_p_ecg_p_12,K_BLOOD,NA_BLOOD,ALT_BLOOD,AST_BLOOD,L_BLOOD,ROE,TIME_B_S,R_AB_1_n,R_AB_3_n,NA_KB,NOT_NA_KB,NITR_S,NA_R_1_n,GEPAR_S_n,RAZRIV,ZSN,REC_IM,DRESSLER)
            #pass prediction to template
            return render_template('predict.html', prediction = prediction)
   
        except ValueError:
            return "Please Enter valid values"
  
        pass
    pass
def preprocessDataAndPredict(AGE,STENOK_AN,FK_STENOK,IBS_POST,ZSN_A,nr_04,S_AD_KBRIG,D_AD_KBRIG,S_AD_ORIT,D_AD_ORIT,K_SH_POST,ant_im,lat_im,ritm_ecg_p_07,n_r_ecg_p_04,n_p_ecg_p_10,
n_p_ecg_p_12,K_BLOOD,NA_BLOOD,ALT_BLOOD,AST_BLOOD,L_BLOOD,ROE,TIME_B_S,R_AB_1_n,R_AB_3_n,NA_KB,NOT_NA_KB,NITR_S,NA_R_1_n,GEPAR_S_n,RAZRIV,ZSN,REC_IM,DRESSLER):
    
    #keep all inputs in array
    test_data = [AGE,STENOK_AN,FK_STENOK,IBS_POST,ZSN_A,nr_04,S_AD_KBRIG,D_AD_KBRIG,S_AD_ORIT,D_AD_ORIT,K_SH_POST,ant_im,lat_im,ritm_ecg_p_07,n_r_ecg_p_04,n_p_ecg_p_10,
n_p_ecg_p_12,K_BLOOD,NA_BLOOD,ALT_BLOOD,AST_BLOOD,L_BLOOD,ROE,TIME_B_S,R_AB_1_n,R_AB_3_n,NA_KB,NOT_NA_KB,NITR_S,NA_R_1_n,GEPAR_S_n,RAZRIV,ZSN,REC_IM,DRESSLER]
    print(test_data)
    
    #convert value data into numpy array
    test_data = np.array(test_data)
    
    #reshape array
    test_data = test_data.reshape(1,-1)
   # print(test_data)
    outFileFolder = 'output/'
    filePath = outFileFolder + 'randomforest_model.pkl'
    #print(filePath)
    #open file
    file = open(filePath,"rb")
   # file = open("C:\\Users\\theam\Documents\\Excelr\\Project\\PMyocardial1\\output\\randomforest_model.pkl","rb")
    #file = open("output/randomforest_model.pkl","rb")
    
    
    
    #load trained model
    trained_model = joblib.load(file)
    
    #predict
    prediction = trained_model.predict(test_data)
    
    return prediction
    
    pass
if __name__ == '__main__':
    app.run(debug=True)