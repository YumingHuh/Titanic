clc
clear
close all

%选取训练集和测试集
trainData=readtable('train.csv');
testData=readtable('test.csv');

%将性别标签转化为数据
trainNumber=size(trainData,1);
sexData=zeros(trainNumber,1);
for i=1:trainNumber
    if trainData{i,5} == "male"
        sexData(i,1)=1;
    elseif trainData{i,5}== "female"
        sexData(i,1)=0;
    end
end
%计算存活的先验概率
survivedLabel=trainData(:,2);
survivedData=table2array(survivedLabel)
survivedProbability=tabulate(survivedData);
P_unsurvived=survivedProbability(1,3)/100
P_survived=survivedProbability(2,3)/100
%计算性别和仓位等级的条件概率,p代表仓位等级，s代表性别，sd代表存活与否
trainPclassData=trainData(:,3);
trainPclass=table2array(trainPclassData)
pssdData=zeros(trainNumber,1)
for j=1:trainNumber
    pssdData(j,1)=100*trainPclass(j,1)+10*sexData(j,1)+survivedData(j,1)
end
pssdP=tabulate(pssdData);
% 如：P_p1_s1_sd0=pssdP(110,3)/100
%%
%使用测试集进行数据测试
%将性别数据转化为数据
testNumber=size(testData,1);
testSex=zeros(testNumber,1);
for m=1:testNumber
    if testData{m,4} == "male"
        testSex(m,1)=1;
    elseif testData{m,4}== "female"
        testSex(m,1)=0;
    end
end
testPclassData=testData(:,2);
testPclass=table2array(testPclassData)
%贝叶斯计算概率并比较,P1表示存活概率，P0表示死亡概率
Pre=zeros(testNumber,1)
for n=1:testNumber
    P1=pssdP(100*testPclass(n,1)+10*testSex(n,1)+1,3)/100*P_survived
    P0=pssdP(100*testPclass(n,1)+10*testSex(n,1)+0,3)/100*P_unsurvived
    if P1>P0
        Pre(n,1)=1
    else Pre(n,1)=0
    end
end
testIDData=testData(:,1);
testID=table2array(testIDData);
Precolumns={'PassengerId','Survived'}
Pre_submission=table(testID,Pre,'VariableNames',Precolumns);
writetable(Pre_submission,'submission.csv'); 

