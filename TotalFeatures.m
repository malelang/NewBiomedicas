%% EXTRACCIÓN DE CARACTERÍSTICAS
% CARACTERISTICAS:
%1. Valor RMS de la señal ECG
%2. Cruces por cero de la señal ECG
%3. Valor RMS de la señal de respiracion A
%4. Integral del valor absoluto de la señal de respiración A
%5. Valor absoluto medio de la señal de respiración A
%6. Cambios en los signos de pendiente de la señal de respiración A
%7. Desaturación de oxigeno SP02
%8. Aumento del ritmo cardíaco.
%9. Valor RMS de la señal de respiracion C
%10. Integral del valor absoluto de la señal de respiración C
%11. Valor absoluto medio de la señal de respiración C
%12. Cambios en los signos de pendiente de la señal de respiración C
%13. Valor RMS de la señal de respiracion N
%14. Integral del valor absoluto de la señal de respiración N
%15. Valor absoluto medio de la señal de respiración N
%16. Cambios en los signos de pendiente de la señal de respiración N

function results = TotalFeatures(TrainingECG,TrainingRespA,TrainingSPO2,TrainingRespC,TrainingRespN,win_size,win_inc,state)
    %Declaramos el tama�o de la ventana
[~,N]=size(TrainingECG);
k = 1;
for j=1:6000:N
    tempECG=TrainingECG(:,(j:j+6000-1));
    tempRespA=TrainingRespA(:,(j:j+6000-1));
    tempRespC=TrainingRespC(:,(j:j+6000-1));
    tempRespN=TrainingRespN(:,(j:j+6000-1));
    [feature1,feature2] = extract_featureECG(tempECG',win_size,win_inc);
    [feature3,feature4,feature5,feature6]=extract_featureResp(tempRespA',win_size,win_inc);
    [feature9,feature10,feature11,feature12]=extract_featureResp(tempRespC',win_size,win_inc);
    [feature13,feature14,feature15,feature16]=extract_featureResp(tempRespN',win_size,win_inc);
    mc1(k,:)=mean(feature1);
    mc2(k,:)=mean(feature2); 
    mc3(k,:)=mean(feature3);
    mc4(k,:)=mean(feature4);
    mc5(k,:)=mean(feature5);
    mc6(k,:)=mean(feature6);
    mc9(k,:)=mean(feature9);
    mc10(k,:)=mean(feature10);
    mc11(k,:)=mean(feature11);
    mc12(k,:)=mean(feature12);
    mc13(k,:)=mean(feature13);
    mc14(k,:)=mean(feature14);
    mc15(k,:)=mean(feature15);
    mc16(k,:)=mean(feature16);
    k=k+1;
end
if(state==1)
    for registro=1:7
        mc7(registro,:) = SPO2Detector(TrainingSPO2(registro,1:N));
        mc8(registro,:) = CaractECG(TrainingECG(registro,1:N));
    end
else
    mc7=SPO2Detector(TrainingSPO2);
    mc8=CaractECG(TrainingECG);
end
mc7=mc7';
mc8=mc8';

%% Organizamos las características
if(state==1)
    c1=[mc1(:,1);mc1(:,2);mc1(:,3);mc1(:,4);mc1(:,5);mc1(:,6);mc1(:,7)];
    c2=[mc2(:,1);mc2(:,2);mc2(:,3);mc2(:,4);mc2(:,5);mc2(:,6);mc2(:,7)];
    c3=[mc3(:,1);mc3(:,2);mc3(:,3);mc3(:,4);mc3(:,5);mc3(:,6);mc3(:,7)];
    c4=[mc4(:,1);mc4(:,2);mc4(:,3);mc4(:,4);mc4(:,5);mc4(:,6);mc4(:,7)];
    c5=[mc5(:,1);mc5(:,2);mc5(:,3);mc5(:,4);mc5(:,5);mc5(:,6);mc5(:,7)];
    c6=[mc6(:,1);mc6(:,2);mc6(:,3);mc6(:,4);mc6(:,5);mc6(:,6);mc6(:,7)];
    c7=[mc7(:,1);mc7(:,2);mc7(:,3);mc7(:,4);mc7(:,5);mc7(:,6);mc7(:,7)];
    c8=[mc8(:,1);mc8(:,2);mc8(:,3);mc8(:,4);mc8(:,5);mc8(:,6);mc8(:,7)];
    c9=[mc9(:,1);mc9(:,2);mc9(:,3);mc9(:,4);mc9(:,5);mc9(:,6);mc9(:,7)];
    c10=[mc10(:,1);mc10(:,2);mc10(:,3);mc10(:,4);mc10(:,5);mc10(:,6);mc10(:,7)];
    c11=[mc11(:,1);mc11(:,2);mc11(:,3);mc11(:,4);mc11(:,5);mc11(:,6);mc11(:,7)];
    c12=[mc12(:,1);mc12(:,2);mc12(:,3);mc12(:,4);mc12(:,5);mc12(:,6);mc12(:,7)];
    c13=[mc13(:,1);mc13(:,2);mc13(:,3);mc13(:,4);mc13(:,5);mc13(:,6);mc13(:,7)];
    c14=[mc14(:,1);mc14(:,2);mc14(:,3);mc14(:,4);mc14(:,5);mc14(:,6);mc14(:,7)];
    c15=[mc15(:,1);mc15(:,2);mc15(:,3);mc15(:,4);mc15(:,5);mc15(:,6);mc15(:,7)];
    c16=[mc16(:,1);mc16(:,2);mc16(:,3);mc16(:,4);mc16(:,5);mc16(:,6);mc16(:,7)];
else
    c1=mc1;
    c2=mc2;
    c3=mc3;
    c4=mc4;
    c5=mc5;
    c6=mc6;
    c7=mc7;
    c8=mc8;
    c9=mc9;
    c10=mc10;
    c11=mc11;
    c12=mc12;
    c13=mc13;
    c14=mc14;
    c15=mc15;
    c16=mc16;
end
%% Las devolvemos en un solo vector
results = [c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 c15 c16];

end