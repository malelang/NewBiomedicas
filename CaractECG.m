%Esta funcion se encarga de ver cuándo hay un aumento en el ritmo cardiaco
%por cada minuto, haciendo uso de la desviación estandar en los picos RR y
%comparando si se desvía mas de 2 veces la desviación estandar. En caso de
%que se cuenten 3 o mas eventos de este tipo en un minuto, se puede decir
%que el ritmo cardíaco ha aumentado
%Entradas: señal que contiene el ECG y frecuencia de muestreo
%Salidas: un vector que contiene un 1 o 0 logico dependiendo de si hubo o
%no hubo aumento en el ritmo cardíaco (respectivamente) en ese minuto
function heartRateArise=CaractECG(FullECG,fs)
    % Señal ECG al cuadrado
    dECG=detrend(FullECG');
    dECG=dECG';
    ECGSquared=abs(dECG).^2;
    [PKSECG,LOCSECG]=findpeaks(ECGSquared,'MinPeakHeight',0.6,'MinPeakDistance',6);
    
    j=1;
    pace=6000;
    cont=0;
    while(pace<=length(ECGSquared))
        liminf=LOCSECG(pace-6000+1<LOCSECG);
        limsup=liminf(liminf<=pace);
        if (isempty(LOCSECG)==1)
            heartRateArise(j)=0;
        else
           RRtime=diff(limsup)/fs;
            stdvtime=std(RRtime);
            difftiempos=diff(RRtime);
            for i=1:length(difftiempos)
                if(abs(difftiempos(i))>(2*stdvtime))
                    cont=cont+1;
                end
            end
            if(cont>=3)
                heartRateArise(j)=1;
            else
                heartRateArise(j)=0;
            end 
        end
        pace=pace+6000;
        j=j+1;
        cont=0;
    end
    
end