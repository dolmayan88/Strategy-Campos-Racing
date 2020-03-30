# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:10:17 2020

@author: Alvaro
"""

######################LIBRARIES IMPORTED#######################################

from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output,State
import plotly.graph_objs as go

###############################################################################

######################CLASSES AND FUNCTIONS####################################

def convert2time(laptime):
    ''' TAB: Format_data
        This function converts a laptime to seconds as a float. The input has to be a string in the form mm:ss.t'''
    try:
        int(laptime)
    except:
        sec=laptime.split('.')[0]
        tenth=float('0.'+laptime.split('.')[1])
        laptime=time.strptime(sec,'%M:%S')
        laptime=datetime.timedelta(hours=laptime.tm_hour, minutes=laptime.tm_min, seconds=laptime.tm_sec).total_seconds()+tenth
    return float(laptime)

def Eq_Model(x,A,B,C):
    
    return A*x+(B*np.exp(C*(x))) 

class DDBB():
    db = create_engine('mysql://mf6bshg8uxot8src:nvd3akv0rndsmc6v@nt71li6axbkq1q6a.cbetxkdyhwsb.us-east-1.rds.amazonaws.com:3306/ss0isbty55bwe8te')
    
    def GetData(self,DDBBColumn,DDBBTable,DDBBColumnFilter,DDBBValueFilter):
        if isinstance(DDBBColumn,list):
            
            DDBB_df = pd.read_sql_query("SELECT " + ",".join(DDBBColumn) + " FROM `" + DDBBTable + "` WHERE `" + DDBBColumnFilter + "` LIKE '" + DDBBValueFilter +"'",self.db)
                     
        else:
            if str.lower(DDBBColumn) == 'all':
                DDBB_df = pd.read_sql_query("SELECT * FROM `" + DDBBTable + "` WHERE `" + DDBBColumnFilter + "` LIKE '" + DDBBValueFilter +"'",self.db)
                
            else:
                DDBB_df = pd.read_sql_query("SELECT " + DDBBColumn + " FROM `" + DDBBTable + "` WHERE `" + DDBBColumnFilter + "` LIKE '" + DDBBValueFilter +"'",self.db)
            
        if len(DDBB_df)>0:
            return DDBB_df
        else:
            print("No hay ningún dato en la BBDD perteneciente a esos filtros.")
       
class Event():

    def __init__(self,Naming_convention,DDBB):
        self.DDBB=DDBB
        self.Name = str(Naming_convention)
        self.Chsip = self.Name.split("_")[0]
        self.Year = self.Name.split("_")[1][0:2]
        self.Track = self.Name.split("_")[1][5:8]
        self.Session = self.Name.split("_")[2]
        self.PdfFlag = int(self.DDBB.GetData("pdf_flag","Calendar","session_id",self.Name).values)
        self.TrackFuelPenalty = float(self.DDBB.GetData("fuel_penalty","Calendar","session_id",self.Name).values)
        self.NrOfDifferentCompoundsUsed = len(np.unique(self.DDBB.GetData("all","TyreAlloc","session",self.Name)[['s1', 's2','s3','s4','s5','s6','s7','s8','s9','s10']].dropna().values))-1
        self.PrimeCompound = "".join(c for c in str(self.DDBB.GetData("Prime_Tyre","Calendar","session_id",self.Name).values) if c.isupper())
        self.OptionCompound = "".join(c for c in str(self.DDBB.GetData("Option_Tyre","Calendar","session_id",self.Name).values) if c.isupper())
        self.DriverList = self.DDBB.GetData("driver","TyreAlloc","session",self.Name)['driver'].unique().tolist() #Now getting the list from tyrealloc table , because in some event, there are less than 20 drivers and it is failing if i choose driverlist from rawtiming or pdftiming with all the drivers
        self.NrofLaps = int(self.DDBB.GetData("lap","PdfTiming","session",self.Name)['lap'].max())
        self.DriverEndPosition = self.DDBB.GetData(["driver","position"],"PdfTiming","session",self.Name)[self.DDBB.GetData("lap","PdfTiming","session",self.Name)['lap'] == self.NrofLaps]
        self.DriverStartPosition =self.DDBB.GetData(["driver","startpos"],"TyreAlloc","session",self.Name)
        self.LapTimesDf = self.DDBB.GetData(["driver","lap","s1","s2","s3","laptime","InPit","Pits"],"PdfTiming","session",self.Name)
        self.LapTimesDf["laptime"]=self.LapTimesDf.laptime.replace('','0:00.000').apply(lambda x: convert2time(x))
        self.LapTimesDf["laptime_fuel_corrected"]=self.LapTimesDf.apply(lambda row: row.laptime + (row.lap-1)*self.TrackFuelPenalty,axis=1)
        self.DriverCompoundDf= self.DDBB.GetData("all","TyreAlloc","session",self.Name)
        if self.NrOfDifferentCompoundsUsed > 1:
            self.LapTimesDf['Tyre_Compound'] = self.LapTimesDf.apply (lambda row: self.label_compound(row), axis=1)
        else:
            self.LapTimesDf['Tyre_Compound'] = self.PrimeCompound
        self.LapTimesDf['Pits']=self.LapTimesDf.apply(lambda row: self.label_pits(row),axis =1)
        self.LapTimesDf['StintLaps']=self.LapTimesDf.groupby(['driver','Pits']).cumcount()+1
        self.LapTimesDf_Prime = self.LapTimesDf[self.LapTimesDf['Tyre_Compound']==self.PrimeCompound]
        self.MaxNrofLaps_Prime = self.LapTimesDf_Prime['StintLaps'].max()
        self.LapTimesDf_Option = self.LapTimesDf[self.LapTimesDf['Tyre_Compound']==self.OptionCompound]
        self.MaxNrofLaps_Option = self.LapTimesDf_Option['StintLaps'].max()
        self.DriverList_Prime = self.LapTimesDf['driver'][self.LapTimesDf['Tyre_Compound']==self.PrimeCompound].unique().tolist()
        self.DriverList_Option = self.LapTimesDf['driver'][self.LapTimesDf['Tyre_Compound']==self.OptionCompound].unique().tolist()

    def label_pits (self,row):

        if row['Pits'] == 0 and row['InPit']==0:
           return int(0)
             
        if row['Pits'] == 1 and row['InPit']==1:
           return int(0)
        if row['Pits'] == 1 and row['InPit']==0:
           return int(1) 
        if row['Pits'] == 2 and row['InPit']==1:
           return int(1)
        if row['Pits'] == 2 and row['InPit']==0:
           return int(2)
        if row['Pits'] == 3 and row['InPit']==1:
           return int(2)
        if row['Pits'] == 3 and row['InPit']==0:
           return int(3) 
        if row['Pits'] == 4 and row['InPit']==1:
           return int(3)
        if row['Pits'] == 4 and row['InPit']==0:
           return int(4)
        if row['Pits'] == 5 and row['InPit']==1:
           return int(4)
        if row['Pits'] == 5 and row['InPit']==0:
           return int(5) 
        if row['Pits'] == 6 and row['InPit']==1:
           return int(5)
        if row['Pits'] == 6 and row['InPit']==0:
           return int(6)
        if row['Pits'] == 7 and row['InPit']==1:
           return int(6)
        if row['Pits'] == 7 and row['InPit']==0:
           return int(7)  
        if row['Pits'] == 8 and row['InPit']==1:
           return int(7)
        if row['Pits'] == 8 and row['InPit']==0:
           return int(8)
        if row['Pits'] == 9 and row['InPit']==1:
           return int(8)
        if row['Pits'] == 9 and row['InPit']==0:
           return int(9)          
        if row['Pits'] == 10 and row['InPit']==1:
           return int(9)
        if row['Pits'] == 10 and row['InPit']==0:
           return int(10)        
        
    def label_compound (self,row):
       for driver in self.DriverList:
           if row['driver'] == driver and row['Pits'] == 0:
               return self.DriverCompoundDf['s1'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]        
           if row['driver'] == driver and row['Pits'] == 1:
               if row['InPit']== 1:                    
                   return self.DriverCompoundDf['s1'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]                  
               else:
                   return self.DriverCompoundDf['s2'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
              
           if row['driver'] == driver and row['Pits'] == 2:
               if row['InPit']== 1:
                   
                   return self.DriverCompoundDf['s2'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0] 
               else:
                   return self.DriverCompoundDf['s3'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
           if row['driver'] == driver and row['Pits'] == 3:
               if row['InPit']== 1:
                   return self.DriverCompoundDf['s3'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.DriverCompoundDf['s4'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
           if row['driver'] == driver and row['Pits'] == 4:
               if row['InPit']== 1:
                   return self.DriverCompoundDf['s4'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.DriverCompoundDf['s5'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
           if row['driver'] == driver and row['Pits'] == 5:
               if row['InPit']== 1:
                   return self.DriverCompoundDf['s5'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.DriverCompoundDf['s6'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
           if row['driver'] == driver and row['Pits'] == 6:
               if row['InPit']== 1:
                   return self.DriverCompoundDf['s6'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.DriverCompoundDf['s7'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
           if row['driver'] == driver and row['Pits'] == 7:
               if row['InPit']== 1:
                   return self.DriverCompoundDf['s7'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.DriverCompoundDf['s8'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
           
           if row['driver'] == driver and row['Pits'] == 8:
               if row['InPit']== 1:
                   return self.DriverCompoundDf['s8'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.DriverCompoundDf['s9'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
           
           if row['driver'] == driver and row['Pits'] == 9:
               if row['InPit']== 1:
                   return self.DriverCompoundDf['s9'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.DriverCompoundDf['s10'][self.DriverCompoundDf['driver']==row['driver']].values.tolist()[0]
                                  
    def TopXDrivers(self,top_number):
        top_driver_list=self.DriverEndPosition['driver'][self.DriverEndPosition['position']<top_number+1].tolist()
        return top_driver_list
    
    def PlotValuesByDriversByMedianByModel(self,laps,driverlist,y_median_values,y_model_values,y_values_mode='deg',comp='',track_sector='all'):
        if str.lower(track_sector) == 'all':
            sector="laptime_fuel_corrected"
        elif str.lower(track_sector) == 's1':
            sector = "s1"
        elif str.lower(track_sector) == 's2':
            sector = "s2"
        elif str.lower(track_sector) == 's3':
            sector = "s3"
        if str.lower(comp) =='prime':
           
            laptimes_df=self.LapTimesDf_Prime
            LapTimesDf_SelectedDrivers=laptimes_df[laptimes_df['driver'].isin(driverlist)]
            GroupByDriver=LapTimesDf_SelectedDrivers.groupby('driver')
            
        elif str.lower(comp) == 'option':
           
            laptimes_df=self.LapTimesDf_Option
            LapTimesDf_SelectedDrivers=laptimes_df[laptimes_df['driver'].isin(driverlist)]
            GroupByDriver=LapTimesDf_SelectedDrivers.groupby('driver')
            
        else:
           
            laptimes_df=self.LapTimesDf
            LapTimesDf_SelectedDrivers=laptimes_df[laptimes_df['driver'].isin(driverlist)]
            GroupByDriver=LapTimesDf_SelectedDrivers.groupby('driver')
            
        if (isinstance(driverlist,str)) and ((str.lower(driverlist) == 'all') or (driverlist == '')):            
            for drivers in self.DriverList:
                if y_values_mode == 'deg':                    
                    y_values=GroupByDriver.get_group(drivers)[sector]-GroupByDriver.get_group(drivers)[sector].min()
                    x_values=GroupByDriver.get_group(drivers)['StintLaps']
                    plt.plot(x_values[:-2],y_values[1:-1])
                else:
                    y_values=GroupByDriver.get_group(drivers)[sector]
                    x_values=GroupByDriver.get_group(drivers)['StintLaps']
                    plt.plot(x_values[:-2],y_values[1:-1])
                    
            
            if isinstance(y_median_values,list):
                plt.plot(laps,y_median_values,marker='o',markersize=8,color='blue')
            else:
                pass
            if isinstance(y_model_values,list):
                plt.plot(laps,y_model_values,marker='x',markersize=8,color='red')
            else:
                pass
            plt.show()    
                
                
        elif isinstance(driverlist,list):
            for drivers in driverlist:
                
                if y_values_mode == 'deg':
                    y_values=GroupByDriver.get_group(drivers)[sector]-GroupByDriver.get_group(drivers)[sector].min()
                    x_values=GroupByDriver.get_group(drivers)['StintLaps']
                    plt.plot(x_values[:-2],y_values[1:-1])
                else:
                    y_values=GroupByDriver.get_group(drivers)[sector]
                    x_values=GroupByDriver.get_group(drivers)['StintLaps']
                    plt.plot(x_values[:-2],y_values[1:-1])
                    
            
            if isinstance(y_median_values,np.ndarray):
                plt.plot(laps,y_median_values,marker='o',markersize=8,color='blue')
            else:
                pass
            if isinstance(y_model_values,np.ndarray):
                plt.plot(laps,y_model_values,marker='x',markersize=8,color='red')
            else:
                pass
            plt.show()     
                
    def GetLaptimesOrDegMedianByDriver(self,laptimes_df,driverlist,y_values_mode='deg',track_sector='all'):
        
        """
       Inputs:
           driverlist -> Here we can enter the filter for the drivers. 
                         It has to be a list of drivers or a string 'all' to take every driver into account.
       
           y_values_mode -> "deg" or "laptime" -> It is calculating the median of
                           all the laptimes or the median of the absolute deg,
                           that means median laptime per lap - minimum median of all the laps (All values greater than zero)
           
        """
        
        "User entering driverlist='all' or ''"
        if str.lower(track_sector) == 'all':
            sector="laptime_fuel_corrected"
        elif str.lower(track_sector) == 's1':
            sector = "s1"
        elif str.lower(track_sector) == 's2':
            sector = "s2"
        elif str.lower(track_sector) == 's3':
            sector = "s3"
            
        if (isinstance(driverlist,str)) and ((str.lower(driverlist) == 'all') or (driverlist == '')):  
            laps=self.NrofLaps
            for drivers in self.DriverList:
                y_median_laptime_values=[laptimes_df[sector][laptimes_df['StintLaps']==laps].median() for laps in range(1,laps+1)]
                y_median_deg_values=y_median_laptime_values-np.nanmin(np.array(y_median_laptime_values))
                x_median_values=list(range(1,laps+1))
                if y_values_mode =='':
                    return x_median_values,y_median_laptime_values
                else:
                    return x_median_values,y_median_deg_values
                
                "User entering driverlist='all' or empty str " 
                
        elif isinstance(driverlist,list): 
            for drivers in driverlist:
                LapTimesDf_SelectedDrivers=laptimes_df[laptimes_df['driver'].isin(driverlist)]
                laps = LapTimesDf_SelectedDrivers['StintLaps'].max()
                y_median_laptime_values=[LapTimesDf_SelectedDrivers[sector][LapTimesDf_SelectedDrivers['StintLaps']==laps].median() for laps in range(1,laps+1)]
                y_median_deg_values=y_median_laptime_values-np.nanmin(np.array(y_median_laptime_values))
                x_median_values=list(range(1,laps+1))
                if y_values_mode =='':
                    return x_median_values,y_median_laptime_values
                else:
                    return x_median_values,y_median_deg_values
                   
    def TyreModelCoeffs(self,laps,laptimes):
        
#        global y_weight
        g=[0.25,0.25,0.25]
#        y_weight = np.empty(len(laptimes))
#        y_weight.fill(1)
#        y_weight[10:] = 0.1
        
        coeff,cov =curve_fit(Eq_Model,laps,laptimes)#,sigma=y_weight,absolute_sigma=True)
    
        return coeff,cov

class Strategy(Event):
    pass

class Results(Event):
    pass

######################DASH APP#################################################



###############################################################################  
    
###############################################################################
                      
   


#####################MAIN PROGRAM##############################################                
#if __name__ == "__main__":
app=dash.Dash()
#Heidi_DDBB=DDBB()
#conn = self.DDBB.db.connect()         
#Event=Event(str(input("Introduce la sesion que quieras!")))
##    DriverList=Event.DriverList
#DriverList=Event.TopXDrivers(10)
    
#    if Event.NrOfDifferentCompoundsUsed == 1:
    #All Drivers Prime
#    laps,y_median_deg=Event.GetLaptimesOrDegMedianByDriver(Event.LapTimesDf,DriverList,y_values_mode="deg",track_sector='all')
#    TyreModelCoeffs,TyreModelCovar=Event.TyreModelCoeffs(laps[:-2],y_median_deg[1:-1])
#    TyreModelCoeffs=TyreModelCoeffs.tolist()
#    y_model_deg=Eq_Model(np.array(laps[:-2]),TyreModelCoeffs[0],TyreModelCoeffs[1],TyreModelCoeffs[2])
#    Event.PlotValuesByDriversByMedianByModel(laps[:-2],DriverList,y_median_deg[1:-1],y_model_deg,y_values_mode="deg",comp="all",track_sector='all')
#    else:
#        #Prime Model
#        prime_laps,y_median_prime_deg=Event.GetLaptimesOrDegMedianByDriver(Event.LapTimesDf_Prime,DriverList,y_values_mode='deg',track_sector='s1')
#        TyreModelCoeffs_prime,TyreModelCovar_Prime=Event.TyreModelCoeffs(prime_laps[:-3],y_median_prime_deg[1:-2])
#        TyreModelCoeffs_prime=TyreModelCoeffs_prime.tolist()
#        y_model_prime_deg=Eq_Model(np.array(prime_laps[:-2]),TyreModelCoeffs_prime[0],TyreModelCoeffs_prime[1],TyreModelCoeffs_prime[2])
#        Event.PlotValuesByDriversByMedianByModel(prime_laps[:-2],DriverList,y_median_prime_deg[1:-1],y_model_prime_deg,y_values_mode="deg",comp='prime',track_sector='s1')
#        #Option Model
#        option_laps,y_median_option_deg=Event.GetLaptimesOrDegMedianByDriver(Event.LapTimesDf_Option,DriverList,y_values_mode="deg",track_sector='s1')
#        TyreModelCoeffs_option,TyreModelCovar_option=Event.TyreModelCoeffs(option_laps[:-2],y_median_option_deg[1:-1])
#        TyreModelCoeffs_option=TyreModelCoeffs_option.tolist()
#        y_model_option_deg=Eq_Model(np.array(option_laps[:-2]),TyreModelCoeffs_option[0],TyreModelCoeffs_option[1],TyreModelCoeffs_option[2])
#        Event.PlotValuesByDriversByMedianByModel(option_laps[:-2],DriverList,y_median_option_deg[1:-1],y_model_option_deg,y_values_mode="deg",comp='option',track_sector='s1')

    
app.layout = html.Div([
                    
                      html.Button(id='button'),         
                      dcc.Graph(id='feature-graphic')
                     
                     ]
    )
@app.callback(Output('feature-graphic','figure'),
              [Input('button','n_clicks')])
def update_graph(n_clicks):
    global DDBB,Event,laps,y_median_deg,trace_all
    Heidi_DDBB=DDBB()
    conn = Heidi_DDBB.db.connect()
    Event=Event("F2_19R08BUD_R2",Heidi_DDBB)
    DriverList=Event.TopXDrivers(10)
    laps,y_median_deg=Event.GetLaptimesOrDegMedianByDriver(Event.LapTimesDf,DriverList,y_values_mode="deg",track_sector='all')
    TyreModelCoeffs,TyreModelCovar=Event.TyreModelCoeffs(laps[:-2],y_median_deg[1:-1])
    TyreModelCoeffs=TyreModelCoeffs.tolist()
    y_model_deg=Eq_Model(np.array(laps[:-2]),TyreModelCoeffs[0],TyreModelCoeffs[1],TyreModelCoeffs[2])
    track_sector='all'
    if str.lower(track_sector) == 'all':
            sector="laptime_fuel_corrected"
    elif str.lower(track_sector) == 's1':
        sector = "s1"
    elif str.lower(track_sector) == 's2':
        sector = "s2"
    elif str.lower(track_sector) == 's3':
        sector = "s3"
    comp=''
    driverlist=DriverList
    if str.lower(comp) =='prime':
       
        laptimes_df=Event.LapTimesDf_Prime
        LapTimesDf_SelectedDrivers=laptimes_df[laptimes_df['driver'].isin(driverlist)]
        GroupByDriver=LapTimesDf_SelectedDrivers.groupby('driver')
        
    elif str.lower(comp) == 'option':
       
        laptimes_df=Event.LapTimesDf_Option
        LapTimesDf_SelectedDrivers=laptimes_df[laptimes_df['driver'].isin(driverlist)]
        GroupByDriver=LapTimesDf_SelectedDrivers.groupby('driver')
        
    else:
       
        laptimes_df=Event.LapTimesDf
        LapTimesDf_SelectedDrivers=laptimes_df[laptimes_df['driver'].isin(driverlist)]
        GroupByDriver=LapTimesDf_SelectedDrivers.groupby('driver')
    
    trace_drivers=[]
    y_values_mode='deg'
    for drivers in driverlist:
            
            if y_values_mode == 'deg':
                y_values=GroupByDriver.get_group(drivers)[sector]-GroupByDriver.get_group(drivers)[sector].min()
                x_values=GroupByDriver.get_group(drivers)['StintLaps']
                trace_drivers.append(go.Scatter(
                                                x=x_values[2:],
                                                y=y_values[2:],
                                                mode='lines+markers',
                                                name=drivers
            
            
        ))
            else:
                y_values=GroupByDriver.get_group(drivers)[sector]
                x_values=GroupByDriver.get_group(drivers)['StintLaps']
                trace_drivers.append(go.Scatter(
                                                x=x_values[2:],
                                                y=y_values[2:],
                                                mode='lines+markers',
                                                name=drivers
            
            
        ))
                
            
                
                
                
    trace_median=go.Scatter(
            x=laps[2:],
            y=y_median_deg[2:],
            mode='lines+markers',
            name='Median of all drivers'
            
            
        )
    trace_model=go.Scatter(
            x=laps[2:],
            y=y_model_deg[2:],
            mode='lines+markers',
            name='Math Model Fit'            
            
        )
    
    trace_drivers.append(trace_median)
    trace_drivers.append(trace_model)
    layout=go.Layout(
            xaxis={'title': 'Laps'},
            yaxis={'title': 'Absolute Deg'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    conn.close()
    Heidi_DDBB.db.dispose()
    return dict(data=trace_drivers,layout=layout)

#print(DriverList)
#print(Event.DriverList_Prime)
#print(Event.DriverList_Option)
#conn.close()
#Heidi_DDBB.db.dispose()

if __name__ == "__main__":
    app.run_server()  

###############################################################################













####################EXTRA INFO#################################################

###crear un grupo
#groupbyprime=Event.LapTimesDf_Prime.groupby(['driver','Pits'])
####groups pinta los filtros del grupo
#for groups in groupbyprime.groups:
#    print(groups)

####pinta el dataframe de uno de los grupos
#groupbyprime.get_group(('KIN',1))    

###Da la longitud del dataframe de cada grupo
#groupbyprime.size()

###Contar algo agrupando
#Event.LapTimesDf['StintLaps']=Event.LapTimesDf.groupby(['driver','Pits']).cumcount()+1
    
###############################################################################