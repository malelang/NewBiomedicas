
function [logicSPO]=SPO2Detector(sp)
    
    j=1;
    pace=50; %cada medio segundo saco una media de los valores
    for i=1:pace:length(sp)
        mediasp(j)=mean(sp(i:i+pace-1));
        j=j+1;
    end
    slope=diff(mediasp);
    slope2=downsample(slope,2);
    slope2=upsample(slope2,2);
    k=0;
    n=1;
    minequivalente=120;
    contador=0;
    while(k<length(slope2))
        signal=slope2(k+1:k+minequivalente);
        [PKSSPO,LOCSSPO]=findpeaks(signal,'MinPeakHeight',0.01);
        intervalos=diff(LOCSSPO);
        for i=1:length(intervalos)
            if(intervalos(i)<5)
                contador=contador+1;
            end        
        end
        if(contador>=3)
            logicSPO(n)=1;
        else
            logicSPO(n)=0;
        end
        k=k+120;
        contador=0;
        n=n+1;
        intervalos=[];
    end   
end