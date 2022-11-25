#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import preprocessing
import numpy as np
import pandas as pd
import random


# In[2]:


all_countries = ['Abkhazia', 'Afghanistan', 'Albania', 'Alderney', 'Algeria',
       'American Samoa', 'Andalusia', 'Andorra', 'Angola', 'Anguilla',
       'Antigua and Barbuda', 'Arameans Suryoye', 'Argentina', 'Armenia',
       'Artsakh', 'Aruba', 'Australia', 'Austria', 'Azerbaijan',
       'Bahamas', 'Bahrain', 'Bangladesh', 'Barawa', 'Barbados',
       'Basque Country', 'Belarus', 'Belgium', 'Belize', 'Benin',
       'Bermuda', 'Bhutan', 'Bolivia', 'Bonaire',
       'Bosnia and Herzegovina', 'Botswana', 'Brazil',
       'British Virgin Islands', 'Brittany', 'Brunei', 'Bulgaria',
       'Burkina Faso', 'Burma', 'Burundi', 'Cambodia', 'Cameroon',
       'Canada', 'Canary Islands', 'Cape Verde', 'Cascadia', 'Catalonia',
       'Cayman Islands', 'Central African Republic', 'Chad',
       'Chagos Islands', 'Chameria', 'Chile', 'China PR', 'Colombia',
       'Comoros', 'Congo', 'Cook Islands', 'Corsica', 'Costa Rica',
       'County of Nice', 'Crimea', 'Croatia', 'Cuba', 'Curaçao', 'Cyprus',
       'Czech Republic', 'Czechoslovakia', 'DR Congo', 'Darfur',
       'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic',
       'East Timor', 'Ecuador', 'Egypt', 'El Salvador', 'Ellan Vannin',
       'England', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini',
       'Ethiopia', 'Falkland Islands', 'Faroe Islands', 'Felvidék',
       'Fiji', 'Finland', 'France', 'French Guiana', 'French Polynesia',
       'Frøya', 'Gabon', 'Galicia', 'Gambia', 'Georgia', 'German DR',
       'Germany', 'Ghana', 'Gibraltar', 'Gotland', 'Gozo', 'Greece',
       'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala',
       'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Găgăuzia',
       'Haiti', 'Hitra', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland',
       'India', 'Indonesia', 'Iran', 'Iraq', 'Iraqi Kurdistan',
       'Isle of Man', 'Isle of Wight', 'Israel', 'Italy', 'Ivory Coast',
       'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kabylia', 'Kazakhstan',
       'Kenya', 'Kernow', 'Kiribati', 'Kosovo', 'Kuwait', 'Kyrgyzstan',
       'Kárpátalja', 'Laos', 'Latvia', 'Lebanon', 'Lesotho', 'Liberia',
       'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macau',
       'Madagascar', 'Madrid', 'Malawi', 'Malaysia', 'Maldives', 'Mali',
       'Malta', 'Martinique', 'Matabeleland', 'Mauritania', 'Mauritius',
       'Mayotte', 'Menorca', 'Mexico', 'Micronesia', 'Moldova', 'Monaco',
       'Mongolia', 'Montenegro', 'Montserrat', 'Morocco', 'Mozambique',
       'Myanmar', 'Namibia', 'Nepal', 'Netherlands',
       'Netherlands Antilles', 'New Caledonia', 'New Zealand',
       'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'North Macedonia',
       'Northern Cyprus', 'Northern Ireland', 'Northern Mariana Islands',
       'Norway', 'Occitania', 'Oman', 'Orkney', 'Padania', 'Pakistan',
       'Palau', 'Palestine', 'Panama', 'Panjab', 'Papua New Guinea',
       'Paraguay', 'Parishes of Jersey', 'Peru', 'Philippines', 'Poland',
       'Portugal', 'Provence', 'Puerto Rico', 'Qatar', 'Raetia',
       'Republic of Ireland', 'Republic of St. Pauli', 'Rhodes',
       'Romani people', 'Romania', 'Russia', 'Rwanda', 'Réunion',
       'Saare County', 'Saint Helena', 'Saint Kitts and Nevis',
       'Saint Lucia', 'Saint Martin', 'Saint Pierre and Miquelon',
       'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sark',
       'Saudi Arabia', 'Scotland', 'Senegal', 'Serbia',
       'Serbia and Montenegro', 'Seychelles', 'Shetland', 'Sierra Leone',
       'Silesia', 'Singapore', 'Sint Maarten', 'Slovakia', 'Slovenia',
       'Solomon Islands', 'Somalia', 'Somaliland', 'South Africa',
       'South Korea', 'South Ossetia', 'South Sudan', 'Soviet Union',
       'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Surrey', 'Sweden',
       'Switzerland', 'Syria', 'Székely Land', 'Sápmi',
       'São Tomé and Príncipe', 'Tahiti', 'Taiwan', 'Tajikistan',
       'Tamil Eelam', 'Tanzania', 'Thailand', 'Tibet', 'Timor-Leste',
       'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
       'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu',
       'Two Sicilies', 'Uganda', 'Ukraine', 'United Arab Emirates',
       'United Koreans in Japan', 'United States',
       'United States Virgin Islands', 'Uruguay', 'Uzbekistan', 'Vanuatu',
       'Vatican City', 'Venezuela', 'Vietnam', 'Wales',
       'Wallis Islands and Futuna', 'Western Armenia', 'Western Isles',
       'Western Sahara', 'Western Samoa', 'Yemen', 'Yemen AR',
       'Yemen DPR', 'Ynys Môn', 'Yorkshire', 'Yugoslavia', 'Zambia',
       'Zanzibar', 'Zaïre', 'Zimbabwe', 'Åland Islands']


# In[3]:


label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(all_countries)


# In[4]:


Group_A= ["Qatar","Ecuador","Senegal","Netherlands"]
Group_B= ["England","Iran","United States","Wales"]
Group_C= ["Argentina","Saudi Arabia","Mexico","Poland"]
Group_D= ["France","Australia","Denmark","Tunisia"]
Group_E= ["Spain","Costa Rica","Germany","Japan"]
Group_F= ["Belgium","Canada","Morocco","Croatia"]
Group_G= ["Brazil","Serbia","Switzerland","Cameroon"]
Group_H= ["Portugal","Ghana","Uruguay","South Korea"]
Groups={"Group A":Group_A,"Group B":Group_B,"Group C":Group_C,"Group D":Group_D,"Group E":Group_E,"Group F":Group_F,"Group G":Group_G,"Group H":Group_H}


# In[5]:


countries = ["Qatar","Ecuador","Senegal","Netherlands","England","Iran","United States","Wales","Argentina","Saudi Arabia","Mexico","Poland",
            "France","Australia","Denmark","Tunisia", "Spain","Costa Rica","Germany","Japan", "Belgium","Canada","Morocco","Croatia",
            "Brazil","Serbia","Switzerland","Cameroon","Portugal","Ghana","Uruguay","South Korea"]


# In[6]:


countries_label = label_encoder.transform(countries)
clabel = pd.DataFrame(zip(countries, countries_label), columns = ['country', 'label'])
clabel['label'] = clabel['label'].astype('category')


# In[7]:


clabel = {name:value for name, value in zip(countries, list(clabel['label']))}


# In[8]:


year = 2022
stadium = "Qatar"
stadium_num = clabel[stadium]
host_num=stadium_num


# In[9]:


def select_winning_team(probability_array):
    prob_lst=[round(probability_array[0][i],3) for i in range(2)]
    if (prob_lst[0]>prob_lst[1]):
        out=0
    elif (prob_lst[0]<prob_lst[1]):
        out=1
    elif (prob_lst[0]==prob_lst[1]):
        out=2
    return out,prob_lst


# In[15]:


# select team group match
def select_team_groupmatch(selectteams, Model):
    select_team_wins_dct={}
    select_goal_scored_dct={}
    select_goal_against_dct={}
    select_win_dct={}
    select_draw_dct={}
    select_lost_dct={}
    for i in range(len(selectteams)):
        j=i+1
        select_team_1 = selectteams[i]
        select_team_wins=0
        while j < len((selectteams)):
            select_team_2=selectteams[j]
            select_team_lst=[select_team_1,select_team_2]
            select_Input_vector=np.array([[year,host_num, clabel[select_team_1] , clabel[select_team_2]]])
            res=Model.predict(select_Input_vector)

            select_win,prob_lst = select_winning_team(res)
            select_goal_scored_dct[select_team_1] = select_goal_scored_dct.get(select_team_1,0)+prob_lst[0]
            select_goal_scored_dct[select_team_2] = select_goal_scored_dct.get(select_team_2,0)+prob_lst[1]

            select_goal_against_dct[select_team_1] = select_goal_against_dct.get(select_team_1,0)+prob_lst[1]
            select_goal_against_dct[select_team_2] = select_goal_against_dct.get(select_team_2,0)+prob_lst[0]

            try:
                if (select_win)==0:
                    select_team_wins_dct[select_team_1] = select_team_wins_dct.get(select_team_1,0)+3
                    select_team_wins_dct[select_team_2] = select_team_wins_dct.get(select_team_2,0)
                    
                    select_win_dct[select_team_1] = select_win_dct.get(select_team_1,0)+1
                    select_win_dct[select_team_2] = select_win_dct.get(select_team_2,0)
                    select_lost_dct[select_team_2] = select_lost_dct.get(select_team_2,0)+1
                    select_lost_dct[select_team_1] = select_lost_dct.get(select_team_1,0)
                    select_draw_dct[select_team_2] = select_draw_dct.get(select_team_2,0)
                    select_draw_dct[select_team_1] = select_draw_dct.get(select_team_1,0)

                elif (select_win)==1:
                    select_team_wins_dct[select_team_2] = select_team_wins_dct.get(select_team_2,0)+3
                    select_team_wins_dct[select_team_1] = select_team_wins_dct.get(select_team_1,0)
                    
                    select_win_dct[select_team_2] = select_win_dct.get(select_team_2,0)+1
                    select_win_dct[select_team_1] = select_win_dct.get(select_team_1,0)
                    select_lost_dct[select_team_1] = select_lost_dct.get(select_team_1,0)+1
                    select_lost_dct[select_team_2] = select_lost_dct.get(select_team_2,0)
                    select_draw_dct[select_team_1] = select_draw_dct.get(select_team_1,0)
                    select_draw_dct[select_team_2] = select_draw_dct.get(select_team_2,0)

            except IndexError:
                select_team_wins_dct[select_team_1] = select_team_wins_dct.get(select_team_1,0)+1
                select_team_wins_dct[select_team_2] = select_team_wins_dct.get(select_team_2,0)+1
                
                select_draw_dct[select_team_1] = select_draw_dct.get(select_team_1,0)+1
                select_draw_dct[select_team_2] = select_draw_dct.get(select_team_2,0)+1
                
                select_win_dct[select_team_1] = select_win_dct.get(select_team_1,0)
                select_lost_dct[select_team_1] = select_lost_dct.get(select_team_1,0)
              
                select_win_dct[select_team_2] = select_win_dct.get(select_team_2,0)
                select_lost_dct[select_team_2] = select_lost_dct.get(select_team_2,0)
                    
            j=j+1
    select_results=[select_win_dct,select_draw_dct,select_lost_dct,select_team_wins_dct,select_goal_scored_dct,select_goal_against_dct]

    return select_results


# In[16]:


##Group stage Matches
def all_group_stage(Model):
    Group_standings={}
    for grp_name in list(Groups.keys()):
        probable_countries=Groups[grp_name]
        team_wins_dct={}
        goal_scored_dct={}
        goal_against_dct={}
        win_dct={}
        draw_dct={}
        lost_dct={}
        for i in range(len(probable_countries)):
            j=i+1
            team_1=probable_countries[i]
            team_wins=0
            while j<len((probable_countries)):
                team_2=probable_countries[j]
                team_lst=[team_1,team_2]
                Input_vector=np.array([[year,stadium_num,clabel[team_1],clabel[team_2]]])
                res=Model.predict(Input_vector)

                win,prob_lst=select_winning_team(res)
                goal_scored_dct[team_1] = goal_scored_dct.get(team_1,0)+prob_lst[0]
                goal_scored_dct[team_2] = goal_scored_dct.get(team_2,0)+prob_lst[1]

                goal_against_dct[team_1] = goal_against_dct.get(team_1,0)+prob_lst[1]
                goal_against_dct[team_2] = goal_against_dct.get(team_2,0)+prob_lst[0]

                try:
                    if (win)==0:
                        team_wins_dct[team_1] = team_wins_dct.get(team_1,0)+3
                        team_wins_dct[team_2] = team_wins_dct.get(team_2,0)
                    
                        win_dct[team_1] = win_dct.get(team_1,0)+1
                        win_dct[team_2] = win_dct.get(team_2,0)
                        lost_dct[team_2] = lost_dct.get(team_2,0)+1
                        lost_dct[team_1] = lost_dct.get(team_1,0)
                        draw_dct[team_2] = draw_dct.get(team_2,0)
                        draw_dct[team_1] = draw_dct.get(team_1,0)

                    elif (win)==1:
                        team_wins_dct[team_2] = team_wins_dct.get(team_2,0)+3
                        team_wins_dct[team_1] = team_wins_dct.get(team_1,0)
                    
                        win_dct[team_2] = win_dct.get(team_2,0)+1
                        win_dct[team_1] = win_dct.get(team_1,0)
                        lost_dct[team_1] = lost_dct.get(team_1,0)+1
                        lost_dct[team_2] = lost_dct.get(team_2,0)
                        draw_dct[team_1] = draw_dct.get(team_1,0)
                        draw_dct[team_2] = draw_dct.get(team_2,0)

                except IndexError:
                    team_wins_dct[team_1] = team_wins_dct.get(team_1,0)+1
                    team_wins_dct[team_2] = team_wins_dct.get(team_2,0)+1
                
                    draw_dct[team_1] = draw_dct.get(team_1,0)+1
                    draw_dct[team_2] = draw_dct.get(team_2,0)+1
                
                    win_dct[team_1] = win_dct.get(team_1,0)
                    lost_dct[team_1] = lost_dct.get(team_1,0)
                
                    win_dct[team_2] = win_dct.get(team_2,0)
                    lost_dct[team_2] = lost_dct.get(team_2,0)
                    
                j=j+1
        group_results=[win_dct,draw_dct,lost_dct,team_wins_dct,goal_scored_dct,goal_against_dct]
        Group_standings[grp_name]=group_results

    return Group_standings


# In[17]:


def round_of_16(Group_standings, Model):
        ##Round of 16 Section_1
        qualified_teams_1=[]
        standings=list(Group_standings.keys())
        i=0
        while i < (len(standings)):
                A_team= sorted(Group_standings[standings[i]][3].items(), key=lambda x: x[1], reverse=True)
                team_1=A_team[0][0]
                B_team= sorted(Group_standings[standings[i+1]][3].items(), key=lambda x: x[1], reverse=True)
                team_2=B_team[1][0]
        
                team_lst=[team_1,team_2]
    
                Input_vector=np.array([[year,host_num,clabel[team_1],clabel[team_2]]])
                res=Model.predict(Input_vector)
                win,_=select_winning_team(res)

                try:
                        qualified_teams_1.append(team_lst[win])
                except IndexError:
                        winning_team=random.choice(team_lst)
                        qualified_teams_1.append(winning_team)
                i=i+2

        ##Round of 16 Section_2
        qualified_teams_2=[]
        standings=list(Group_standings.keys())
        i=0
        while i < (len(standings)):
                A_team= sorted(Group_standings[standings[i]][3].items(), key=lambda x: x[1], reverse=True)
                team_1=A_team[1][0]
                B_team= sorted(Group_standings[standings[i+1]][3].items(), key=lambda x: x[1], reverse=True)
                team_2=B_team[0][0]
    
                team_lst=[team_1,team_2]
    
                Input_vector=np.array([[year,host_num,clabel[team_1],clabel[team_2]]])
                res=Model.predict(Input_vector)
                win,_=select_winning_team(res)

                try:
                        qualified_teams_2.append(team_lst[win])
            
                except IndexError:
                        winning_team=random.choice(team_lst)
                        qualified_teams_2.append(winning_team)
                i=i+2
                
        return qualified_teams_1, qualified_teams_2


# In[18]:


#Quarter Finals

def quarter(qualified_teams_1, qualified_teams_2, Model):
    Semifinal_teams=[]
    i=0
    while i < (len(qualified_teams_1))-1:
        team_1= qualified_teams_1[i]
        team_2= qualified_teams_1[i+1]
    
        team_lst=[team_1,team_2]

        Input_vector=np.array([[year,host_num,clabel[team_1],clabel[team_2]]])
        res=Model.predict(Input_vector)
        win,_=select_winning_team(res)

        try:
            Semifinal_teams.append(team_lst[win])
            
        except IndexError:
            winning_team=random.choice(team_lst)
            Semifinal_teams.append(winning_team)
        i=i+2
    
    i=0
    while i < (len(qualified_teams_2))-1:
        team_1= qualified_teams_2[i]
        team_2= qualified_teams_2[i+1]

        team_lst=[team_1,team_2]
    
        Input_vector=np.array([[year,host_num,clabel[team_1],clabel[team_2]]])
        res=Model.predict(Input_vector)
        win,_=select_winning_team(res)

        try:
            Semifinal_teams.append(team_lst[win])
            
        except IndexError:
            winning_team=random.choice(team_lst)
            Semifinal_teams.append(winning_team)
        i=i+2

    return Semifinal_teams


# In[19]:


#Semi Finals

def semi_final(Semifinal_teams, Model):
    final_teams=[]
    third_place_match_teams=[]
    i=0
    while i < (len(Semifinal_teams))-1:
        team_1= Semifinal_teams[i]
        team_2= Semifinal_teams[i+1]
    
        team_lst=[team_1,team_2]
    
        Input_vector=np.array([[year,host_num,clabel[team_1],clabel[team_2]]])
        res=Model.predict(Input_vector)
        win,_=select_winning_team(res)

        try:
            final_teams.append(team_lst[win])
            third_place_match_teams.append(team_lst[(win+1)%2])

            
        except IndexError:
            winning_team=random.choice(team_lst)
            final_teams.append(winning_team)
            team_lst.remove(winning_team)
            third_place_match_teams.append(team_lst[0])
        i=i+2
    
    return final_teams, third_place_match_teams


# In[20]:


#Third Place match

def third_place(third_place_match_teams, Model):
    team_1= third_place_match_teams[1]
    team_2= third_place_match_teams[0]
    
    team_lst=[team_1,team_2]
    
    Input_vector=np.array([[year,host_num,clabel[team_1],clabel[team_2]]])
    res=Model.predict(Input_vector)
    win,_=select_winning_team(res)

    try:
        place_3=team_lst[win]
            
    except IndexError:
        winning_team=random.choice(team_lst)
        place_3 = winning_team
    
    return place_3


# In[21]:


# finals

def final(final_teams, Model):
    team_1= final_teams[1]
    team_2= final_teams[0]
    
    team_lst=[team_1,team_2]
    
    Input_vector=np.array([[year,host_num,clabel[team_1],clabel[team_2]]])
    res=Model.predict(Input_vector)
    win,_=select_winning_team(res)

    try:
        winner=team_lst[win]
        place_2=team_lst[(win+1)%2]
            
    except IndexError:
        winning_team=random.choice(team_lst)
        winner=winning_team
    
        team_lst.remove(winning_team)
        place_2=team_lst[0]
    
    return winner, place_2 


def two_teams_result(team_1, team_2, Model):
    Input_vector=np.array([[year,stadium_num,clabel[team_1],clabel[team_2]]])
    res=Model.predict(Input_vector)
    return res[0][0], res[0][1] # team_1 vs team_2


