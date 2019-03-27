function salida = caract_voz(x,Fs)
% Entradas: x -> se�al de voz, 
%           Fs -> Frecuencia de muestreo
%% Caracter�sticas del espectro completo de la se�al
% Frecuencia promedio y Mediana del espectro completo de la se�al
%[mediaF,medianF,~,~] = centerfreq(Fs,x); % Calcula la frecuencia media donde el espectro de la se�al completa tiene su mayor potencia
%% Características de las frecuencias melódicas de la voz
% Estas características se obtienen luego de aplicar la FFT a un conjunto
% de ventanas móviles sobre la señal de voz, para detectar los "tonos" en
% la señal de voz
[Frequency_amp,Frequency,FunFrequency] = PreProcessing(x,Fs); 
% Realiza el ventaneo de Haming, devuelve las frecuencias centrales,
% potencia de promedio y frecuencias fundamentales de cada espectro de una ventana
%% Desviación estándar
sd = std(Frequency);
%% Primer cuantil
Q25 = quantile(Frequency,0.25);
%% Cuarto cuantil
Q75 = quantile(Frequency,0.75);
%% Rango intercuantil
IQR = iqr(Frequency);
%% Asimetr�a
skew = skewness(Frequency);
%% Kurtosis
kurt = kurtosis(Frequency);
%% Entrop�a espectral
% l=log(Frequency_amp);
% vind=isinf(l);
% for i=1:length(vind)
%     if vind(i)==1
%         l(i)=0;
%     end
% end
% spent = -sum(Frequency_amp.*l)./log(length(Frequency));
%% Achatamiento del espectro
sfm = geomean(Frequency)/mean(Frequency);
%% Moda de la frecuancia
modfrec = mode(Frequency);
%% Las siguientes caracter�sticas se extraen con base en las frecuencias fundamentales (tonos) de la voz
% %% Frecuancia fundamental promedio
% meanfun = mean(FunFrequency);
% %% Frecuencia fundamental m�nima
% minfun = min(FunFrequency);
% %% Frecuancia fundamental m�xima
% maxfun = max(FunFrequency);

salida = [sd,Q25,Q75,IQR,skew,kurt,sfm,modfrec];


