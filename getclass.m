
function class_training = getclass(registro)
switch(registro)
    case 1
        class_training = ones(240,1);
        class_training(1:13) = 0;
        class_training(227:230) = 0;
    case 2
        class_training = ones(240,1);
        class_training(1:21) = 0;
        class_training(130:175) = 0;
        class_training(201:206) = 0;
    case 3
        class_training = ones(240,1);
        class_training(1:13) = 0;
        class_training(60:104) = 0;
        class_training(147:193) = 0;
        class_training(229:239) = 0;
    case 4
        class_training = ones(240,1);
        class_training(1:34) = 0;
        class_training(175:176) = 0;
        class_training(147:193) = 0;
        class_training(229:239) = 0;
    case 5
        class_training=zeros(240,1);
    case 6
        class_training=zeros(240,1);
        class_training(21)=1;
    case 7
        class_training=zeros(240,1);
    case 8
        class_training=zeros(240,1);
        class_training(5)=1;
        class_training(135)=1;
        class_training(159:162)=1;
end

end    

