
function [feature1,feature2,feature3,feature4] = extract_featureResp(data,win_size,win_inc)

if nargin < 3
    if nargin < 2
        win_size = 256;
    end
    win_inc = 32;
end

[Ndata,Nsignal] = size(data);

%valor medio de la señal
medio=mean(data);
tresh=medio.*0.02;
umbral=max(tresh);


feature1 = getrmsfeat(data,win_size,win_inc);
feature2 = getiavfeat(data,win_size,win_inc);
% ar_order = 4;
% feature2 = getarfeat(data,ar_order,win_size,win_inc);
feature3 = getmavfeat(data,win_size,win_inc);
%feature4 = getzcfeat(data,umbral,win_size,win_inc);
feature4 = getsscfeat(data,umbral,win_size,win_inc);


end