function [vrms,zcr]=caractiempo(sx)

th = hilbert(sx); 
env = abs(th); 
vrms=rms(env);

zcr = sum(abs(diff(sx>0)))/length(sx);

%ALE: vrms=0.0183 .   zcr= 0.1870
%SEBAS: vrms=0.0144 .   zcr=0.2179
end