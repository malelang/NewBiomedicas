function [filename] = ExportToExcel(nombres,vrms,zcr,mc1,mc2,mc3,mc4,mc5,mc6,mc7,mc8,mc9,mc10,mc11,mc12,mc13,mc14,mc15,mc16)
filename = 'Feasures.xlsx';
A = {'Descripcion','Nombre_Archivo','ValorRMS','TasaZRC','Frec_Prom','Frec_mediana','Desv_Estandar','Primer_cuantil','Tercer_Cuartil','Rango_intercuantil','Asimetria ','Curtosis','Entropia_espectral','Achatamiento_espectro','Moda','Frec_Media','Frec_Minima','Frec_maxima'};
sheet = 1;
xlRange = 'A1';
xlswrite(filename, A, sheet, xlRange)
k=1;
for i=1:20
    for j=1:7
        Nombres(k) = nombres(i,j);
        ValorRms(k) = mc1(i,j);
        Vzcr(k) = mc2(i,j);
        s1(k) = mc3(i,j); 
        s2(k) = mc4(i,j); 
        s3(k) = mc5(i,j); 
        s4(k) = mc6(i,j); 
        s5(k) = mc7(i,j); 
        s6(k) = mc8(i,j); 
        s7(k) = mc9(i,j); 
        s8(k) = mc10(i,j); 
        s9(k) = mc11(i,j); 
        s10(k) = mc12(i,j); 
        s11(k) = mc13(i,j); 
        s12(k) = mc14(i,j); 
        s13(k) = mc15(i,j); 
        s14(k) = mc16(i,j);    
        k = k+1;
    end  
end  
 xlswrite(filename, Nombres(:), 1, 'B2')
 xlswrite(filename, ValorRms(:), 1, 'C2')
 xlswrite(filename, Vzcr(:), 1, 'D2')
 xlswrite(filename, s1(:), 1, 'E2')
 xlswrite(filename, s2(:), 1, 'F2')
 xlswrite(filename, s3(:), 1, 'G2')
 xlswrite(filename, s4(:), 1, 'H2')
 xlswrite(filename, s5(:), 1, 'I2')
 xlswrite(filename, s6(:), 1, 'J2')
 xlswrite(filename, s7(:), 1, 'K2')
 xlswrite(filename, s8(:), 1, 'L2')
 xlswrite(filename, s9(:), 1, 'M2')
 xlswrite(filename, s10(:), 1, 'N2')
 xlswrite(filename, s11(:), 1, '02')
 xlswrite(filename, s12(:), 1, 'P2')
 xlswrite(filename, s13(:), 1, 'Q2')
 xlswrite(filename, s14(:), 1, 'R2')
 

end