#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import fasttext
import pandas as pd
import random
import json
import os
import sys

dir_route = sys.argv[1]

datas = pd.read_csv(dir_route + "/train.txt", sep="\t", names=["value"])
datas = datas.sample(frac=1)

train = datas.iloc[0:int(datas.shape[0] * 0.8)]
test = datas.iloc[int(datas.shape[0] * 0.8) + 1:]

train_output = open(dir_route + "/train_output.txt", "w+")

for c1, c2 in train.iterrows():
    value = c2["value"]
    values = json.loads(value)
    rr = values['content'].strip(",").strip(" ").split(",")
    random.shuffle(rr)
    train_output.writelines("__label__" + str(values["gender"]) + "\t" + " ".join(rr) + "\n")
train_output.close()

classifier = fasttext.train_supervised(dir_route + '/train_output.txt', label='__label__', wordNgrams=4, epoch=20,
                                       lr=0.1, dim=100)

count = 0
female_count = 0
male_count = 1
correct = 0
female_correct_count = 0
male_correct_count = 0
female_sum = 0
female_correct_sum = 0
male_sum = 0
male_correct_sum = 0
for c1, c2 in test.iterrows():
    count += 1
    value = c2["value"]
    values = json.loads(value)
    result = (classifier.predict(" ".join(values["content"].strip(",").strip(" ").split(","))))
    if values["gender"] == 2:
        female_count += 1
        female_sum += result[1][0]
    if values["gender"] == 1:
        male_count += 1
        male_sum += result[1][0]

    if "__label__" + str(values["gender"]) == result[0][0]:
        correct += 1
        if values["gender"] == 2:
            female_correct_count += 1
            female_correct_sum += result[1][0]
        if values["gender"] == 1:
            male_correct_count += 1
            male_correct_sum += result[1][0]

accuracy = correct * 1.0 / count
female_accuracy = female_correct_count * 1.0 / female_count
male_accuracy = male_correct_count * 1.0 / male_count
print  ("new model accuracy: all " + str(accuracy) + " female " + str(female_accuracy) + " male " + str(male_accuracy))

if accuracy >= 0.93:
    classifier.save_model(dir_route + "/fast_text_gender_model.freeze")
else:
    if os.path.exists(dir_route + "/fast_text_gender_model.freeze"):
        classifier = fasttext.load_model("fast_text_gender_model.freeze")
    else:
        printf("error:no model!")
# 对比是否更新模型
"""
import os
if not os.path.exists("fast_text_gender_model.freeze"):
    classifier.save_model("fast_text_gender_model.freeze")
else :
    old_classifier = fasttext.load_model("fast_text_gender_model.freeze")
    count = 0 
    female_count = 0
    male_count=1
    correct = 0
    female_correct_count= 0
    male_correct_count = 0
    female_sum = 0
    female_correct_sum = 0
    male_sum = 0
    male_correct_sum = 0
    for c1,c2 in test.iterrows():
        count += 1

        result =  (old_classifier.predict(" ".join(c2["content"].strip(",").strip(" ").split(","))))
        if c2["gender"]==2:
            female_count += 1
            female_sum += result[1][0]
        if c2["gender"]==1:
            male_count += 1
            male_sum += result[1][0]

        if "__label__"+str(c2["gender"]) == result[0][0]:
            correct += 1 
            if c2["gender"]==2:
                female_correct_count += 1
                female_correct_sum += result[1][0]
            if c2["gender"]==1:
                male_correct_count += 1
                male_correct_sum += result[1][0]



    old_accuracy = correct*1.0/count
    old_female_accuracy = female_correct_count*1.0/female_count
    old_male_accuracy = male_correct_count*1.0/male_count 
    print  ("old model accuracy: all "+str(old_accuracy) + " female "+str(old_female_accuracy) + " male " + str(old_male_accuracy))

    if old_accuracy<=accuracy:
        classifier.save_model("fast_text_gender_model.freeze")
    else :
        classifier = old_classifier
"""

# 预测集

predict_datas = pd.read_csv(dir_route + "/predict.txt", sep="\t", names=["value"])
predict_datas = predict_datas.fillna("")
result_file = open(dir_route + "/predict_result.csv", "w+")
result_file.writelines("user_uuid,gender" + "\n")
i = 0
for c1, c2 in predict_datas.iterrows():
    i = i + 1
    value = c2["value"]
    values = json.loads(value)
    result = (classifier.predict(" ".join(values["content"].strip(",").strip(" ").split(","))))
    if len(values["user_uuid"]) == 0 or result[1][0] <= 0.8:
        continue
    if i == len(predict_datas):
        result_file.writelines(values["user_uuid"] + "," + result[0][0].split("__")[2])
    else:
        result_file.writelines(values["user_uuid"] + "," + result[0][0].split("__")[2] + "\n")
result_file.close()











