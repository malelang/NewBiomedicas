%% ESTA FUNCIÓN HACE LA PREPARACIÓN DE LOS DATOS DE TESTING EN UN PRDATASET
function [DatasetTotal]=preparacion1(feature_testing,class_testing)

lista_feasures = char('VRMSecg','ZCecg','RMSrespA','IAVrespA','MAVrespA','SSCrespA','SP02','HRAV','RMSrespC','IAVrespC','MAVrespC','SSCrespC','RMSrespN','IAVrespN','MAVrespN','SSCrespN');
DatasetTotal = prdataset(feature_testing, class_testing,'featlab',lista_feasures);
end