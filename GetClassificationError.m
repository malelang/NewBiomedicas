function E=GetClassificationError(class_testing,LABEL)
    error=zeros(1,length(class_testing));
    for i=1:length(class_testing)
        if(class_testing(i)~=LABEL(i))
            error(i)=1;
        end
    end

    E=10*sum(error)/length(class_testing);
    fprintf('El error de clasificación es: %d \n',E);

    for i=1:length(LABEL)
        clas1=LABEL(LABEL==1);
        clas2=LABEL(LABEL==0);
        real1=class_testing(class_testing==1);
        real2=class_testing(class_testing==0);
    end

    ec1=100*abs((sum(real1)-sum(clas1)))/length(real1);
    ec2=100*abs((sum(real2)-sum(clas2)))/length(real2);

    fprintf('El error de clasificación del movimiento 1 es: %d \n',ec1);
    fprintf('El error de clasificación del movimiento 2 es: %d \n',ec2);
end