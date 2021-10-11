clc
clear
close all

%ѡȡѵ�����Ͳ��Լ�
trainData=readtable('train.csv');
testData=readtable('test.csv');

%���Ա��ǩת��Ϊ����
trainNumber=size(trainData,1);
sexData=zeros(trainNumber,1);
for i=1:trainNumber
    if trainData{i,5} == "male"
        sexData(i,1)=1;
    elseif trainData{i,5}== "female"
        sexData(i,1)=0;
    end
end
%��������������
survivedLabel=trainData(:,2);
survivedData=table2array(survivedLabel)
survivedProbability=tabulate(survivedData);
P_unsurvived=survivedProbability(1,3)/100
P_survived=survivedProbability(2,3)/100
%�����Ա�Ͳ�λ�ȼ�����������,p�����λ�ȼ���s�����Ա�sd���������
trainPclassData=trainData(:,3);
trainPclass=table2array(trainPclassData)
pssdData=zeros(trainNumber,1)
for j=1:trainNumber
    pssdData(j,1)=100*trainPclass(j,1)+10*sexData(j,1)+survivedData(j,1)
end
pssdP=tabulate(pssdData);
% �磺P_p1_s1_sd0=pssdP(110,3)/100
%%
%ʹ�ò��Լ��������ݲ���
%���Ա�����ת��Ϊ����
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
%��Ҷ˹������ʲ��Ƚ�,P1��ʾ�����ʣ�P0��ʾ��������
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

