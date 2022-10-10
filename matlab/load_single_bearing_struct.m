addpath('Datensatz');
cd Datensatz
Datensatz_dir=dir();
a=1;
g=0;
for n=3:length(Datensatz_dir)
    bearing_num=Datensatz_dir(n).name;
    cd (bearing_num)
    bearing_dir=dir('*.mat');
    
    if bearing_num=='KB24'|bearing_num=='KI16'
        damage_extent=3;
    end
    

    if bearing_num=='KA03'|bearing_num=='KA06'|bearing_num=='KA08'|bearing_num=='KA09'|bearing_num=='KI07'|bearing_num=='KI08'|bearing_num=='KA16'|bearing_num=='KB23'|bearing_num=='KI18'
        damage_extent=2;
    end
       
       
    if bearing_num=='KA04'|bearing_num=='KA15'|bearing_num=='KA22'|bearing_num=='KA30'|bearing_num=='KB27'|bearing_num=='KI04'|bearing_num=='KI14'|bearing_num=='KI17'|bearing_num=='KI21'|bearing_num=='KA01'|bearing_num=='KA05'|bearing_num=='KA07'|bearing_num=='KI01'|bearing_num=='KI03'|bearing_num=='KI05'
        damage_extent=1;
    end
    
    
    if bearing_num=='KB23'|bearing_num=='KB24'|bearing_num=='KB27'
        ir_damage=1;
        or_damage=1;
    end
    
    
    if bearing_num==bearing_num=='KI04'|bearing_num=='KI14'|bearing_num=='KI16'|bearing_num=='KI17'|bearing_num=='KI18'|bearing_num=='KI21'|bearing_num=='KI01'|bearing_num=='KI03'|bearing_num=='KI05'|bearing_num=='KI07'|bearing_num=='KI08'
        ir_damage=1;
        or_damage=0;
    end
    
    
    if bearing_num=='KA04'|bearing_num=='KA15'|bearing_num=='KA16'|bearing_num=='KA22'|bearing_num=='KA30'|bearing_num=='KA01'|bearing_num=='KA03'|bearing_num=='KA05'|bearing_num=='KA06'|bearing_num=='KA07'|bearing_num=='KA08'|bearing_num=='KA09'
        or_damage=1;
        ir_damage=0;
    end
     
    
    if bearing_num=='K001'|bearing_num=='K002'|bearing_num=='K003'|bearing_num=='K004'|bearing_num=='K005'|bearing_num=='K006'
        damage_type='healthy'; 
        or_damage=0;
        ir_damage=0;
        damage_extent=0;
    end
      
    
    if bearing_num=='KA04'|bearing_num=='KA15'|bearing_num=='KA16'|bearing_num=='KA22'|bearing_num=='KA30'|bearing_num=='KB23'|bearing_num=='KB24'|bearing_num=='KB27'|bearing_num=='KI04'|bearing_num=='KI14'|bearing_num=='KI16'|bearing_num=='KI17'|bearing_num=='KI18'|bearing_num=='KI21'
        damage_type='accerlerated lifetime test';
    end
    
    if bearing_num=='KA01'|bearing_num=='KA03'|bearing_num=='KA05'|bearing_num=='KA06'|bearing_num=='KA07'|bearing_num=='KA08'|bearing_num=='KA09'|bearing_num=='KI01'|bearing_num=='KI03'|bearing_num=='KI05'|bearing_num=='KI07'|bearing_num=='KI08'
        damage_type='artificial';
    end
    
        for m=1:(length(bearing_dir))
        new_data_struct(a).bearing_num=bearing_num;
        
        if strfind(bearing_dir(m).name,'N09') == 1
            new_data_struct(a).rpm=900;
        elseif strfind(bearing_dir(m).name,'N15') == 1
            new_data_struct(a).rpm=1500;
        end
        
        if strfind(bearing_dir(m).name,'M07')== 5
        %if strncmpi(bearing_dir(m).name,'M07',20)
            new_data_struct(a).torque=0.7;
        elseif strfind(bearing_dir(m).name,'M01')== 5
            %strncmpi(bearing_dir(m).name,'M01',20)
            new_data_struct(a).torque=0.1;
        end
   
         if strfind(bearing_dir(m).name,'F10')== 9
             %strncmpi(bearing_dir(m).name,'F10',20)
            new_data_struct(a).radial_force=1000;
         elseif strfind(bearing_dir(m).name,'F04')== 9
             %strncmpi(bearing_dir(m).name,'F04',20)
            new_data_struct(a).radial_force=400;
         end
        
         if strcmp(bearing_dir(m).name(end-6),'_')
            mesurement_num=str2num(bearing_dir(m).name(end-5:end-4));
            new_data_struct(a).mesurement=mesurement_num;
         else 
            mesurement_num=str2num(bearing_dir(m).name(end-4));
            new_data_struct(a).mesurement=mesurement_num;
         end
        
         filename=bearing_dir(m).name(1:end-4);
         file=load(filename + ".mat");
         
         new_data_struct(a).vibration_data=file.(filename).Y(7).Data;
         
         
         new_data_struct(a).damage_type=damage_type;
         new_data_struct(a).or_damage=or_damage;
         new_data_struct(a).ir_damage=ir_damage;
         new_data_struct(a).damage_extent=damage_extent;

        a=a+1;
        end
    cd ..
    cd ..
    cd struct_data
    filename= strcat(bearing_num,'_struct.mat');
    save(filename,'new_data_struct'); 
    clear new_data_struct a 
    a = 1;
    cd ..
    cd Datensatz
end
