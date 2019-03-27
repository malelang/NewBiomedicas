%% PROCESAMIENTO DE SEÑALES RESPIRATORIAS
function results=CaractResp(RespSignal,Fs)
    BreathA = RespSignal.^2;
    [RRespA,TRRespA]=findpeaks(BreathA,Fs,'MinPeakHeight',0.4,'MinPeakDistance',35);
    findpeaks(BreathA,Fs,'MinPeakHeight',0.4,'MinPeakDistance',35);
    RRintervalRespA = diff(TRRespA); % 203 x 1, 203 intervalos almacenados
    % 1 segundos equivalen a 100 hz * 10 seg = 1000 muestras
    meanRRintervalRespA = mean(RRintervalRespA);
    aux = 0;
    for p = 1:length(RRintervalRespA)
        if RRintervalRespA(p) > 10 
            ApneaRespA = 1;
            aux = [aux ApneaRespA];
        else 
            ApneaRespA = 0;
            aux = [aux ApneaRespA];
        end
    end
    ApneaEventsWithRespA = aux;
    ApneaEventsA = length(ApneaEventsWithRespA(ApneaEventsWithRespA==1));
   % disp(['Apnea events are: ',num2str(ApneaEventsA),' in this dataset'])
    
    % figure
    % plot(BreathA((1:end),1))
    % hold on
    % %plot(TRRespA,RRespA,'v')
    % hold on
    % plot(ana01er,BreathA(ana01er),'k*')
end