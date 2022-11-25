from operator import mod
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import import_ipynb
import select_team_result as select_team
from sklearn import preprocessing
import pickle
import os
import warnings
os.chdir('C:/Users/haekyu/Desktop/Python/semi')

warnings.filterwarnings('ignore')
Group_A= ["Qatar","Ecuador","Senegal","Netherlands"]
Group_B= ["England","Iran","United States","Wales"]
Group_C= ["Argentina","Saudi Arabia","Mexico","Poland"]
Group_D= ["France","Australia","Denmark","Tunisia"]
Group_E= ["Spain","Costa Rica","Germany","Japan"]
Group_F= ["Belgium","Canada","Morocco","Croatia"]
Group_G= ["Brazil","Serbia","Switzerland","Cameroon"]
Group_H= ["Portugal","Ghana","Uruguay","South Korea"]
Groups={"Group A":Group_A,"Group B":Group_B,"Group C":Group_C,"Group D":Group_D,"Group E":Group_E,"Group F":Group_F,"Group G":Group_G,"Group H":Group_H}

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/prediction")
def predict():
    selectteam = request.args.get("team")
    if selectteam in Group_A:
        selectteams = Group_A
    elif selectteam in Group_B:
        selectteams = Group_B
    elif selectteam in Group_C:
        selectteams = Group_C
    elif selectteam in Group_D:
        selectteams = Group_D
    elif selectteam in Group_E:
        selectteams = Group_E
    elif selectteam in Group_F:
        selectteams = Group_F
    elif selectteam in Group_G:
        selectteams = Group_G
    elif selectteam in Group_H:
        selectteams = Group_H

    Group_standings = select_team.all_group_stage(model)

    select_group_result = select_team.select_team_groupmatch(selectteams, model)
    group_to_16 = sorted(select_group_result[3].items(), key = lambda x: x[1], reverse=True)[0:2]
    group_to_16 = list(zip(*group_to_16))[0]

    if selectteam not in group_to_16:
        endselect = '팀은 조별 경기에서 떨어졌습니다.'
    else:
        qualified_teams_1, qualified_teams_2 = select_team.round_of_16(Group_standings, model)
        if selectteam not in qualified_teams_1 + qualified_teams_2:
            endselect = "팀의 최종성적은 16강입니다"
        else:
            Semifinal_teams = select_team.quarter(qualified_teams_1, qualified_teams_2, model)
            if selectteam not in Semifinal_teams:
                endselect = "팀은 최종성적은 8강입니다"
            else:
                final_teams, third_place_match_teams = select_team.semi_final(Semifinal_teams, model)
                if selectteam not in final_teams:
                    endselect = "팀은 4위입니다"
                    place_3 = select_team.third_place(third_place_match_teams, model)
                    if place_3 == selectteam:
                        endselect = "팀은 3위입니다"
                else:
                    winner, place_2 = select_team.final(final_teams, model)
                    if winner == selectteam:
                        endselect = "팀은 우승입니다"
                    else:
                        endselect = "팀은 2위입니다"
    
    return render_template('generic.html', team = selectteam, endselect = endselect, selectteams =selectteams)

@app.route("/match")
def match():
    return render_template('match.html')
    
@app.route("/matchresult")
def matchresults():
    team = request.args.getlist("matchteam")
    team_1 = team[0]
    team_2 = team[1]
    t1_score, t2_score = select_team.two_teams_result(team_1,team_2, model)
    return render_template('matchresults.html', team = team, t1_score =t1_score , t2_score=t2_score)
    
app.run(host = "0.0.0.0", debug = True)