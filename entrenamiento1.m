function [DatasetTotal,PC,Ws,W,Ap]=entrenamiento1(feature_training,class_training)
%% ENTRENAMIENTO DE LOS CLASIFICADORES
%Variables de salida:
% Dataset Total: lista de los datos junto con las características y las 
% clases de cada registro, para los datos de entrenamiento
% PC: componentes principales
% Ws: Atributos seleccionados
% W: Conjunto de clasificadores
% E: Error mínimo

%Creamos la lista de características
% CARACTERISTICAS:
%1. Valor RMS de la señal ECG
%2. Cruces por cero de la señal ECG
%3. Valor RMS de la señal de respiracion
%4. Integral del valor absoluto de la señal de respiración
%5. Valor absoluto medio de la señal de respiración
%6. Cambios en los signos de pendiente de la señal de respiración
%7. Desaturación de oxigeno SP02
%8. Aumento del ritmo cardíaco.
lista_feasures = char('VRMSecg','ZCecg','RMSrespA','IAVrespA','MAVrespA','SSCrespA','SP02','HRAV','RMSrespC','IAVrespC','MAVrespC','SSCrespC','RMSrespN','IAVrespN','MAVrespN','SSCrespN');

%Creación del dataset
DatasetTotal = prdataset(feature_training, class_training,'featlab',lista_feasures);

%Selección de características

 No_feasures = 8;                                        % Atributos Seleccionados
 [Ws,Rs] = featselm(DatasetTotal,'maha-s','ind',No_feasures); % Feasure Selection map
 DatasetTotal = DatasetTotal*Ws;                          % Map According to Attibutes selected
 DatasetTotal.featlab
 
%% PARTE VI: SELECCION DATOS DE ENTRENAMIENTO Y DE PRUEBA
randreset;
[Ae,Ap,IL,IU] = gendat(DatasetTotal,0.75); 
figure
scatterd(DatasetTotal);
PC = pcam(Ae,0.99999);
figure
scatterd(DatasetTotal*PC);   
        
%% PARTE VII: ENTRENAMIENTO DE LOS CLASIFICADORES
w1 = nmc(Ae*PC); 
w2 = knnc(Ae*PC,3);
w3 = ldc(Ae*PC);        %linear
w4 = qdc(Ae*PC);        % quadratic
w5 = parzenc(Ae*PC);    % Parzen
w6 = dtc(Ae*PC);        % decision tree
%w7 = svc(Ae*PC,proxm('p',6));
w8 = neurc(Ae*PC);      % Neuronal
%w9 = adaboostc(Ae*PC,perlc([],1),50,[],1); %AdaBoost

% Compute and display errors
% Store classifiers in a cell
W = {w1,w2,w3,w4,w5,w6,w8};
name_classf = {'nmc','knnc','ld','qdc','parzenc','dtc','neurc'};
[E,C]=testc(Ap*PC*W); % E, error, C{i} # errores de clasificaci�n en cada clase. Ver tambien confmat  
disp('Error de los clasificadores')
disp(E)
disp('Error minimo')
minE = min(E);
disp(minE)
IminE = find(E==minE);
disp('Mejores Clasificadores')  % Nombres de los clasificadores con m�nimo error de clasificaci�n
nombremejorclasificador=name_classf(IminE);
disp(nombremejorclasificador)

end