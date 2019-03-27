clc;
close all;
clear all;
addpath('/Users/alejandralandinez/Documents/MATLAB/mcode');
addpath('/Users/alejandralandinez/Documents/MATLAB/prtools/prtools');
%% Se usara la base de datos apnea-ecg, bajamos todas las señales de 
%entrenamiento, las cuales son 8. Estas vienen con toda la informacion, y
%tiene los siguientes canales:
%Respiratorios (A,C,N), cardiacos (ECG), y pulsioximetría(SP02)

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
N =1440000; %declaro esto para sacar Dos millones de  muestras equivalentes a 8 horas


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

%% Normalizacion minmax
for i=1:8
    FullECG(i,:)= (ECG(i,:)-min(ECG(i,:)))./(max(ECG(i,:))-min(ECG(i,:)));    
    FullrespC(i,:) = (respC(i,:)-min(respC(i,:)))./(max(respC(i,:))-min(respC(i,:)));
    FullrespA(i,:) = (respA(i,:)-min(respA(i,:)))./(max(respA(i,:))-min(respA(i,:)));
    FullrespN(i,:) = (respN(i,:)-min(respN(i,:)))./(max(respN(i,:))-min(respN(i,:)));
    FullSPO2(i,:)  = (SPO2(i,:)-min(SPO2(i,:)))./(max(SPO2(i,:))-min(SPO2(i,:)));
end

%% Separación de los datos de entrenamiento y de los datos de prueba
TrainingECG=[FullECG(1,:);FullECG(3:8,:)];
TrainingRespA=[FullrespA(1,:);FullrespA(3:8,:)];
TrainingRespC=[FullrespC(1,:);FullrespC(3:8,:)];
TrainingRespN=[FullrespN(1,:);FullrespN(3:8,:)];
TrainingSPO2=[FullSPO2(1,:);FullSPO2(3:8,:)];

%% EXTRACCIÓN DE CARACTERÍSTICAS
%Hacemos la extracción en una función aparte 
win_size = 256;
win_inc = 128; % El solapamiento de la ventana es del 50% en entrenamiento

feature_training=TotalFeatures(TrainingECG,TrainingRespA,TrainingSPO2,TrainingRespC,TrainingRespN,win_size,win_inc);

%% Sacamos las clases siguiendo las anotaciones
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

%% HACEMOS EL ENTRENAMIENTO DE LOS CLASIFICADORES CON LAS CARACTERISTICAS
%QUE POSEEMOS

[Data_training,PC_training,Ws_training,W_training,Ap_training] = entrenamiento1(feature_training,class_training);