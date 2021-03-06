%% CLASIFICACIÓN DE LOS DATOS DE APNEA-ECG
% En este script se hace la clasificación de 2 tipos de apnea, para
% los cuales se usara la base de datos apnea-ecg de physionet, bajamos 
% todas las señales de entrenamiento, las cuales son 8. Estas vienen con 
% toda la informacion, y tiene los siguientes canales:
% Respiratorios (A,C,N), cardiacos (ECG), y pulsioximetría(SP02). 
% En la primera parte identificaremos el dataset de entrenamiento y sacaremos las
% características y lista de clases las cuales usaremos para alimentar el
% modelo de entrenamiento en la herramienta prtools. Posteriormente
% usaremos cualquier otra de las señales como testing.

clc;
close all;
clear all;
addpath('/Users/alejandralandinez/Documents/MATLAB/mcode');
addpath('/Users/alejandralandinez/Documents/MATLAB/prtools/prtools');
%% I PARTE :ORGANIZACION DE LA BASE DE DATOS EN SEÑALES INDIVIDUALES

names=cell(8,1);
for i=1:4
    string=strcat({'a0'},int2str(i),{'er'});
    names{i}=string;
end
for i=1:3
    string=strcat({'c0'},int2str(i),{'er'});
    names{i+4}=string;
end
names{end}='b01er';
N =1440000; %declaro esto para sacar 1'440.000 muestras, equivalentes a 4h.
for i=1:8
    BaseDatos= strcat({'apnea-ecg/'},names{i});
    siginfo = wfdbdesc(BaseDatos{1});
    [signal,fs,tm]= rdsamp(BaseDatos{1}, [1 2 3 4 5], N);
    ECG(:,i)=signal(:,1);
    respA(:,i)=signal(:,2);
    respC(:,i)=signal(:,3);
    respN(:,i)=signal(:,4);
    SPO2(:,i)=signal(:,5);
end

ECG=ECG';
respA=respA';
respC=respC';
respN=respN';
SPO2=SPO2';


t1m=(1:6000)/fs;
t30s=(1:3000)/fs;
t30m=(1:30000)/fs;
t1h=(1:60000)/fs;

% Hacemos una normalización por minimos y maximos
for i=1:8
    FullECG(i,:)= (ECG(i,:)-min(ECG(i,:)))./(max(ECG(i,:))-min(ECG(i,:)));    
    FullrespC(i,:) = (respC(i,:)-min(respC(i,:)))./(max(respC(i,:))-min(respC(i,:)));
    FullrespA(i,:) = (respA(i,:)-min(respA(i,:)))./(max(respA(i,:))-min(respA(i,:)));
    FullrespN(i,:) = (respN(i,:)-min(respN(i,:)))./(max(respN(i,:))-min(respN(i,:)));
    FullSPO2(i,:)  = (SPO2(i,:)-min(SPO2(i,:)))./(max(SPO2(i,:))-min(SPO2(i,:)));
end

%% PARTE II: EXTRACCIÓN DE LOS DATOS DE ENTRENAMIENTO
TrainingECG=[FullECG(1,:);FullECG(3:8,:)];
TrainingRespA=[FullrespA(1,:);FullrespA(3:8,:)];
TrainingRespC=[FullrespC(1,:);FullrespC(3:8,:)];
TrainingRespN=[FullrespN(1,:);FullrespN(3:8,:)];
TrainingSPO2=[FullSPO2(1,:);FullSPO2(3:8,:)];

%% PARTE III: EXTRACCIÓN DE LOS DATOS DE PRUEBA

win_inctesting=32;
TestingECG=FullECG(2,:);
TestingRespA=FullrespA(2,:);
TestingRespC=FullrespC(2,:);
TestingRespN=FullrespN(2,:);
TestingSPO2=FullSPO2(2,:);

% Segunda mitad de la señal
[signal,fs,tm]= rdsamp('apnea-ecg/a02er', [1 2 3 4 5], 2880000, 1440001);
ECG=signal(:,1);
respA=signal(:,2);
respC=signal(:,3);
respN=signal(:,4);
SPO2=signal(:,5);
ECG=ECG';
respA=respA';
respC=respC';
respN=respN';
SPO2=SPO2';
TestingECG= (ECG-min(ECG))./(max(ECG)-min(ECG));    
TestingRespC = (respC-min(respC))./(max(respC)-min(respC));
TestingRespA = (respA-min(respA))./(max(respA)-min(respA));
TestingRespN = (respN-min(respN))./(max(respN)-min(respN));
TestingSPO2  =(SPO2-min(SPO2))./(max(SPO2)-min(SPO2));


%% PARTE IV: EXTRACCIÓN DE CARACTERÍSTICAS DE LOS DATOS DE ENTRENAMIENTO
%Hacemos la extracción en una función aparte 
win_size = 256;
win_inc = 128; % El solapamiento de la ventana es del 50% en entrenamiento

feature_training=TotalFeatures(TrainingECG,TrainingRespA,TrainingSPO2,TrainingRespC,TrainingRespN,win_size,win_inc,1);

%% PARTE V: EXTRACCIÓN DE CARACTERISTICAS DE LOS DATOS DE PRUEBA

feature_testing=TotalFeatures(TestingECG,TestingRespA,TestingSPO2,TestingRespC,TestingRespN, win_size, win_inctesting,2);
%% PARTE VI: EXTRACCIÓN DE LAS CLASES DE LOS DATOS DE ENTRENAMIENTO
%teniendo en cuenta que si registro es igual a:
% 1: a01er
% 2: a02er
% 3: a03er
% 4: a04er
% 5: c01er
% 6: c02er
% 7: c03er
% 8: b01er

class_training=[getclass(1);getclass(3);getclass(4);getclass(5);getclass(6);getclass(7);getclass(8)];

%% PARTE VII: EXTRACCIÓN DE LAS CLASES DE LOS DATOS DE PRUEBA

class_testing=getclass(2);

% Segunda mitad de la señal

class_training=ones(240,1);
class_training(26:29)=0;
class_training(42:47)=0;
class_training(55:58)=0;
class_training(68:71)=0;
class_training(86:88)=0;
class_training(149:151)=0;
class_training(192:195)=0;
%% PARTE VIII: EXTRACCIÓN DEL DATASET DE ENTRENAMIENTO Y ENTRENAMIENTO DE 
% LOS CLASIFICADORES.

[Data_training,PC_training,Ws_training,W_training,Ap_training] = entrenamiento1(feature_training,class_training);
name_classf = {'nmc','knnc','ld','qdc','parzenc','dtc','neurc'};
[E,C]=testc(Ap_training*PC_training*W_training);
minE = min(E);
IminE = find(E==minE);
minimoerror=IminE(1);
nombremejorclasificador=name_classf(IminE);
disp('---------------------------------------------------')

%% PARTE IX: EXTRACCIÓN DEL DATASET DE LOS DATOS DE PRUEBA

[Data_testing] = preparacion1(feature_testing,class_testing);

%% PARTE X: CLASIFICACIÓN DE LOS DATOS DE PRUEBA Y DETERMINACION DEL ERROR

LABEL = labeld(Data_testing,Ws_training*PC_training*W_training{minimoerror});

GetClassificationError(class_testing,LABEL);