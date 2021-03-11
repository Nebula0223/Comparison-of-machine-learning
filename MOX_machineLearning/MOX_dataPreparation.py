"""
The original dataset cannot be used in the machine learning directly
So I analyze it and create a new CSV file
The new CSV file only include the datas that are useful
"""
import xlwt
# MOX_file for training or validating
count=0
MOX_file_name=["./archive/20160930_203718.csv","./archive/20161001_231809.csv","./archive/20161003_085624.csv",
               "./archive/20161004_104124.csv","./archive/20161005_140846.csv","./archive/20161006_182224.csv",
               "./archive/20161007_210049.csv","./archive/20161008_234508.csv","./archive/20161010_095046.csv",
               "./archive/20161011_113032.csv","./archive/20161013_143355.csv","./archive/20161014_184659.csv",
               "./archive/20161016_053656.csv"]
createCSV_file=xlwt.Workbook(encoding='utf-8')
sheet=createCSV_file.add_sheet('MOX Conclusion')
sheet.write(0,0,'CO_Concentration')
sheet.write(0,1,'Humidity')
sheet.write(0,2,'R1_Up_Slope')
sheet.write(0,3,'R1_Down_Slope')
sheet.write(0,4,'R2_Up_Slope')
sheet.write(0,5,'R2_Down_Slope')
sheet.write(0,6,'R3_Up_Slope')
sheet.write(0,7,'R3_Down_Slope')
sheet.write(0,8,'R4_Up_Slope')
sheet.write(0,9,'R4_Down_Slope')
sheet.write(0,10,'R5_Up_Slope')
sheet.write(0,11,'R5_Down_Slope')
sheet.write(0,12,'R6_Up_Slope')
sheet.write(0,13,'R6_Down_Slope')
sheet.write(0,14,'R7_Up_Slope')
sheet.write(0,15,'R7_Down_Slope')
sheet.write(0,16,'R8_Up_Slope')
sheet.write(0,17,'R8_Down_Slope')
sheet.write(0,18,'R9_Up_Slope')
sheet.write(0,19,'R9_Down_Slope')
sheet.write(0,20,'R10_Up_Slope')
sheet.write(0,21,'R10_Down_Slope')
sheet.write(0,22,'R11_Up_Slope')
sheet.write(0,23,'R11_Down_Slope')
sheet.write(0,24,'R12_Up_Slope')
sheet.write(0,25,'R12_Down_Slope')
sheet.write(0,26,'R13_Up_Slope')
sheet.write(0,27,'R13_Down_Slope')
sheet.write(0,28,'R14_Up_Slope')
sheet.write(0,29,'R14_Down_Slope')
write_row=1 # record the row for next writing in CSV

for current_file in MOX_file_name:
    count+=1
    print("file ", count, " is being analyzed……")
    MOX_file=open(current_file,'r')
    lines=MOX_file.readlines()
    MOX_file.close()
    data_row=[]
    data_column=[]
    available_CO_concentration_value=['0','2.22','4.44','6.67','8.89','11.11','13.33','15.56','17.78','20']
    valid_data_begin=0
    newPart_begin_place=[]
    CO_concentration_data=[] # record the CO concentration
    Humidity_data=[] # record the humidity
    SlopeUp_data=[[],[],[],[],[],[],[],[],[],[],[],[],[],[]] # record the MOX resistance(up part) relative to CO centration and humidity
    SlopeDown_data = [[],[],[],[],[],[],[],[],[],[],[],[],[],[]] # record the MOX resistance(down part) relative to CO centration and humidity

    for line in lines:
        data_row.append(line.split(','))
    # get the place of new part
    current_newPart_begin_place=valid_data_begin
    temp_newPart=current_newPart_begin_place
    for col in data_row:
        data_column.append(col[1])
    current_CO=data_column[current_newPart_begin_place]
    while 1:
        temp_newPart+=1
        if temp_newPart==len(lines):
            break
        current_newPart_CO=data_column[temp_newPart]
        if current_newPart_CO!=current_CO and current_newPart_CO in available_CO_concentration_value:
            newPart_begin_place.append(temp_newPart)
            current_CO=current_newPart_CO
    data_column.clear()
    newPart_begin_place.pop()# exclude the unfinished data in the end of the experiment
    del(newPart_begin_place[0])# exclude the preparation data before the experiment

    for i in newPart_begin_place:
        print("current row: ",i)
        # record the current CO concentration
        for col in data_row:
            data_column.append(col[1])
        CO_concentration_data.append(data_column[i])
        data_column.clear()
        # record the current humidity
        for col in data_row:
            data_column.append(col[2])
        Humidity_data.append(data_column[i])
        data_column.clear()

        for R_choice in range(6,20): #14 MOX resistance, column from 6 to 19
            # find the R_Min (the first R_value(<0.1) in a new part)

            # The following are for the up part

            for col in data_row:
                data_column.append(col[R_choice])
            R_Min = data_column[i]
            temp_place_RMin=i
            while 1:
                if float(R_Min)<0.2:
                    break
                else:
                    R_Min=data_column[temp_place_RMin+1]
                    temp_place_RMin+=1
            data_column.clear()
            # find the Time_begin
            for col in data_row:
                data_column.append(col[0])
            Time_begin = data_column[temp_place_RMin]
            data_column.clear()
            # find the R_Max
            for col in data_row:
                data_column.append(col[R_choice])
            R_Max = R_Min
            temp_place = temp_place_RMin
            while 1:
                temp_place+=1
                temp_value = data_column[temp_place]
                if float(temp_value) > float(R_Max) or float(temp_value) < 20: # the setting of 50 is to exclude the situation of the tiny decreace when R_value is still small
                    R_Max=temp_value
                else:
                    break
            data_column.clear()
            # find the Time_end
            for col in data_row:
                data_column.append(col[0])
            Time_end = data_column[temp_place-1]
            data_column.clear()
            # calculate the straight-up slope
            #print(R_Max,R_Min,Time_end,Time_begin)
            slope=round((float(R_Max)-float(R_Min))/(float(Time_end)-float(Time_begin)),2)
            SlopeUp_data[R_choice-6].append(slope)

            #The following are for the down part

            #find the new Time_begin
            new_Time_begin=Time_end
            #find the new R_Max
            new_R_Max=R_Max
            #find the new R_Min
            for col in data_row:
                data_column.append(col[R_choice])
            new_R_Min = new_R_Max
            new_temp_place = temp_place-1
            while 1:
                new_temp_place+=1
                new_temp_value = data_column[new_temp_place]
                if float(new_temp_value) < float(new_R_Min) or float(new_temp_value) > 0.2: # the setting of 0.1 is to exclude the situation of the tiny decreace when R_value is still big
                    new_R_Min=new_temp_value
                else:
                    break
            data_column.clear()
            #find the new Time_end
            for col in data_row:
                data_column.append(col[0])
            new_Time_end = data_column[new_temp_place-1]
            data_column.clear()
            # calculate the straight-down slope
            #print(new_R_Max,new_R_Min,new_Time_begin,new_Time_end)
            new_slope = round((float(new_R_Min) - float(new_R_Max)) / (float(new_Time_end) - float(new_Time_begin)), 2)
            SlopeDown_data[R_choice-6].append(new_slope)

    for write_num in range(0,len(CO_concentration_data)):
        sheet.write(write_row, 0, CO_concentration_data[write_num])
        sheet.write(write_row, 1, Humidity_data[write_num])
        sheet.write(write_row, 2, str(SlopeUp_data[0][write_num]))
        sheet.write(write_row, 3, str(SlopeDown_data[0][write_num]))
        sheet.write(write_row, 4, str(SlopeUp_data[1][write_num]))
        sheet.write(write_row, 5, str(SlopeDown_data[1][write_num]))
        sheet.write(write_row, 6, str(SlopeUp_data[2][write_num]))
        sheet.write(write_row, 7, str(SlopeDown_data[2][write_num]))
        sheet.write(write_row, 8, str(SlopeUp_data[3][write_num]))
        sheet.write(write_row, 9, str(SlopeDown_data[3][write_num]))
        sheet.write(write_row, 10, str(SlopeUp_data[4][write_num]))
        sheet.write(write_row, 11, str(SlopeDown_data[4][write_num]))
        sheet.write(write_row, 12, str(SlopeUp_data[5][write_num]))
        sheet.write(write_row, 13, str(SlopeDown_data[5][write_num]))
        sheet.write(write_row, 14, str(SlopeUp_data[6][write_num]))
        sheet.write(write_row, 15, str(SlopeDown_data[6][write_num]))
        sheet.write(write_row, 16, str(SlopeUp_data[7][write_num]))
        sheet.write(write_row, 17, str(SlopeDown_data[7][write_num]))
        sheet.write(write_row, 18, str(SlopeUp_data[8][write_num]))
        sheet.write(write_row, 19, str(SlopeDown_data[8][write_num]))
        sheet.write(write_row, 20, str(SlopeUp_data[9][write_num]))
        sheet.write(write_row, 21, str(SlopeDown_data[9][write_num]))
        sheet.write(write_row, 22, str(SlopeUp_data[10][write_num]))
        sheet.write(write_row, 23, str(SlopeDown_data[10][write_num]))
        sheet.write(write_row, 24, str(SlopeUp_data[11][write_num]))
        sheet.write(write_row, 25, str(SlopeDown_data[11][write_num]))
        sheet.write(write_row, 26, str(SlopeUp_data[12][write_num]))
        sheet.write(write_row, 27, str(SlopeDown_data[12][write_num]))
        sheet.write(write_row, 28, str(SlopeUp_data[13][write_num]))
        sheet.write(write_row, 29, str(SlopeDown_data[13][write_num]))
        write_row+=1
    print("CO_concentration_data num: ",len(CO_concentration_data))
    print("Humidity_data num: ",len(Humidity_data))
    print("Slope_data num: ",len(SlopeUp_data))
    print("CO_concentration_data: ",CO_concentration_data)
    print("Humidity_data: ",Humidity_data)
    print("file ",count," finished\n")
print("Preparing the excel……")
createCSV_file.save("./MOX Conclusion.csv")
print("Excel finished")