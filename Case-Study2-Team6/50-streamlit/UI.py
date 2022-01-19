# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:19:37 2021

@author: DaronG
"""
## open pkl file
## grab and print row
## run model with inputs from single row
## output results

import streamlit as st
import numpy as np
import pandas as pd
import pickle 

#pickle_in = open('../40-model/xgboost_model.pkl', 'rb') 
#classifier = pickle.load(pickle_in)

def ui_prediction(pitcher_id,
                  batter_id,
                  outs,
                  prev_pitch_type,
                  batter_count,
                  base_runners):
    import pandas as pd
#     df = pd.read_csv("../99-data/cm_vision.csv.zip")
#     drop_cols = ['ab_id', 'on_1b', 'on_2b', 'on_3b', 'batter_fname', 'batter_lname','pitcher_fname', 'pitcher_lname', 'pitch_num', 'pitcher_count']
#     df = df.drop(drop_cols, axis=1)
#     df = df.rename(columns={'Base Runners': 'base_runners'})
#     # clean data
#     for col in ['pitch_type', 'prev_pitch_type', 'batter_count', 'base_runners','outs']:
#         df[col] = df[col].astype('category')
    
#     df['outs'] = df['outs'].cat.codes
#     df['pitch_type'] = df['pitch_type'].cat.codes
#     df['prev_pitch_type'] = df['prev_pitch_type'].cat.codes
#     df['batter_count'] = df['batter_count'].cat.codes
#     df['base_runners'] = df['base_runners'].cat.codes
    df = pd.read_csv("../99-data/ui_df.csv.zip")
    
        # convert input to categorical variables
    def runners_conv(base_runners):
        switcher={
            "Bases empty":0,
            "Runner on base":1
        }
        return switcher.get(base_runners, "nothing")
    base_runners=runners_conv(base_runners)

    ################

    def bcount_conv(batter_count):
        switcher={
            "even":2,
            "behind":1,
            "ahead":0
        }
        return switcher.get(batter_count, "nothing")
    batter_count=bcount_conv(batter_count)

    ###############

    def prev_pitch_conv(prev_pitch):
        switcher={
            "fastball":3,
            "1st_pitch":0,
            "breakingball":1,
            "changeups":2,
            "rare":4
        }
        return switcher.get(prev_pitch, "nothing")
    prev_pitch_type=prev_pitch_conv(prev_pitch_type)
    
        # select data
    input_df = df[(df['pitcher_id'] == pitcher_id) &
                  (df['batter_id'] == batter_id) &
                  (df['outs'] == outs) &
                  (df['prev_pitch_type'] == prev_pitch_type) &
                  (df['batter_count'] == batter_count) &
                  (df['base_runners'] == base_runners)
                 ]
    id_cols = ['pitcher_id', 'batter_id']
    input_df = input_df.drop(id_cols, axis=1)
    
        # establish inputs and outputs
    y = input_df.loc[:,'pitch_type']
    x = input_df.loc[:,input_df.columns != 'pitch_type']

    import pickle
    # save the model to disk
    filename = '../40-model/xgboost_model.pkl'
    # # load the model from disk
    xgboost = pickle.load(open(filename, 'rb'))
    y_pred = xgboost.predict(x)
    
    
    def pred_conv(y_pred):
        switcher={
            2:"fastball",
            0:"breakingball",
            1:"changeups",
            3:"rare"
        }
        return switcher.get(y_pred, "nothing")
    return (pred_conv(y_pred[0]))

def main():
    st.title("Predicting Pitch Type")
    
    pId = st.selectbox("Insert pitcher ID", [477132])
    bId = st.selectbox("Insert batter ID", [457763])
    curr_outs = st.selectbox("Outs", [0,1,2])
    prev_pitchtype = st.selectbox("Previous Pitch",
                                  ["fastball", "breakingball", "changeups", "rare"])
    count = st.selectbox("Count", ["even", "behind", "ahead"])
    ROB = st.selectbox("Runners on Base?", ["Bases empty", "Runner on base"])
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        pitch_prediction = ui_prediction(pitcher_id = int(pId),
                  batter_id = int(bId),
                  outs = curr_outs,
                  prev_pitch_type = prev_pitchtype,
                  batter_count = count,
                  base_runners = ROB)
        st.markdown("---")
        st.markdown(f'**{pitch_prediction.title()}**')
        
        if pitch_prediction == "breakingball":
            st.markdown('''In baseball, a breaking ball is a pitch that 
                        does not travel straight as it approaches the batter; 
                        it will have sideways or downward motion on it, 
                        sometimes both (see slider). A breaking ball is not a 
                        specific pitch by that name, but is any pitch that 
                        "breaks", such as a curveball, slider, or screwball.
                        ''')
        



if __name__ == '__main__':
    main()
