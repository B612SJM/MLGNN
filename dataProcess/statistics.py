#coding=utf-8
import sqlite3
import requests
from bs4 import BeautifulSoup as bs
import math
import pandas as pd
import time
conn=sqlite3.connect("DrugList.db")
cur=conn.cursor()
drugs = pd.read_sql('select * from miner;', conn)
target = list(drugs['target'])
enzyme = list(drugs['enzyme'])
carrier = list(drugs['carrier'])
transporter = list(drugs['transporter'])
drugid=list(drugs['id'])
length=len(target)
cout_t=cout_e=cout_c=cout_tr=0
for i in range(length):
    if target[i]=='':
        cout_t+=1
    if enzyme[i]=='':
        cout_e+=1
    if carrier[i]=='':
        cout_c+=1
    if transporter[i]=='':
        cout_tr+=1
    if target[i]=='' and enzyme[i]==''and carrier[i]==''and transporter[i]=='':
        print("全部没有")
        print(drugid[i])
    if target[i]=='' and enzyme[i]==''and carrier[i]=='':
        print("前三个都没有")
    if target[i]=='' and enzyme[i]==''and transporter[i]=='':
        print("前俩个加最后一个没有")

print("总个数为：%d\n"%(length))
print("target特征存在率%.4f  %d/%d\n"%(float((length-cout_t)/length),length-cout_t,length),
      "enzyme特征存在率%.4f  %d/%d\n" % (float((length - cout_e) / length),length-cout_e,length),
      "carrier特征存在率%.4f  %d/%d\n" % (float((length - cout_c) / length),length-cout_c,length),
      "transporter特征存在率%.4f  %d/%d\n" % (float((length - cout_tr) / length),length-cout_tr,length))
a=1111
str0="times:%d------------------"%(a)
str1 = 'Test ROC score: {:.5f}\n'
str2 = 'Test AP score: {:.5f}\n'
str3 = 'Test F1 score: {:.5f}\n'
str4 = 'Test ACC score: {:.5f}\n\n'
with open('统计记录_test.txt','w') as f:
    f.write(str0)
    f.write(str1)
    f.write(str2)
    f.write(str3)
    f.write(str4)

with open('统计记录_test.txt','a+') as f:
    f.write(str0)
    f.write(str1)
    f.write(str2)
    f.write(str3)
    f.write(str4)