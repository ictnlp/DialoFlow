import json
import os
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from flow_score import *

FLOW_SCORE = FlowScore(MODEL_PATH)

for f in os.listdir("data/"):
    human_dialogues = []
    data_human = json.load(open("data/" + f))
    for i in range(len(data_human)):
        if len(human_dialogues) > 0 and data_human[i]["context"] == human_dialogues[-1]:
            continue
        else:
            human_dialogues.append(data_human[i]["context"])
            a = [k.replace("User: ", "").replace("System: ", "") for k in
                 data_human[i]["context"].split("\n")[:-1]]
            data_human[i]["flow"] = FLOW_SCORE.score(a)
    json.dump(data_human, open("results_Flow/" + f, "w"))

files = os.listdir("results_Flow/")

sort_human_h = []
sort_flow_h = []
for f in files:
    human_scores = []
    FLOW_score = []
    human_dialogues = []
    temp_score = []
    data_human = json.load(open("results_Flow/" + f))
    for i in range(len(data_human)):
        if len(human_dialogues) > 0 and data_human[i]["context"] == human_dialogues[-1]:
            temp_score.append(data_human[i]["human (overall)"])
        else:
            if len(temp_score) > 0:
                human_scores.append(sum(temp_score) / len(temp_score))
            human_dialogues.append(data_human[i]["context"])
            temp_score = [data_human[i]["human (overall)"]]
        if "flow" in data_human[i].keys():
            FLOW_score.append(data_human[i]["flow"])
    human_scores.append(sum(temp_score) / len(temp_score))

    FLOW_score_temp = []
    human_score_temp = []
    for i in range(len(human_dialogues)):
        if len(human_dialogues[i].split("\n")) > 0:
            FLOW_score_temp.append(FLOW_score[i])
            human_score_temp.append(human_scores[i])
    sort_human_h.append(sum(human_score_temp) / len(human_score_temp))
    sort_flow_h.append(sum(FLOW_score_temp) / len(FLOW_score_temp))

print("pearsonr:",pearsonr(sort_human_h, sort_flow_h))
print(spearmanr(sort_human_h, sort_flow_h))
for i in sorted(zip(sort_human_h, sort_flow_h), key=lambda x: x[0]):
    print(i)
    
