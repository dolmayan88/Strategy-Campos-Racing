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
global Event, Heidi_DDBB
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

#def Event_Class_Returner(Event):
#    return Event

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
        self.LapTimesDf['Stint']=self.LapTimesDf['Pits']+1
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
                    
    def GetLaptimesOrDegMedianByDriver(self,laps_filter,compound,driverlist,y_values_mode,track_sector):
        
        """
       Inputs:
           driverlist -> Here we can enter the filter for the drivers. 
                         It has to be a list of drivers or a string 'all' to take every driver into account.
       
           y_values_mode -> "deg" or "laptime" -> It is calculating the median of
                           all the laptimes or the median of the absolute deg,
                           that means median laptime per lap - minimum median of all the laps (All values greater than zero)
           
        """
        
        "User entering driverlist='all' or ''"
        
        if str.lower(track_sector) == 'full lap':
            sector="laptime_fuel_corrected"
        elif str.lower(track_sector) == 's1':
            sector = "s1"
        elif str.lower(track_sector) == 's2':
            sector = "s2"
        elif str.lower(track_sector) == 's3':
            sector = "s3"
            
        if str.lower(comp) == 'prime':
            laptimes_df=User_Event.LapTimesDf_Prime
        elif str.lower(comp) == 'option':
            laptimes_df=User_Event.LapTimesDf_Option
        else:
            pass
           
        
        if (isinstance(driverlist,str)) and ((str.lower(driverlist) == 'all') or (driverlist == '')):              
            for drivers in self.DriverList:
                y_median_laptime_values=[laptimes_df[sector][laptimes_df['StintLaps']==lap].median() for lap in laps_filter]
                y_median_deg_values=y_median_laptime_values-np.nanmin(np.array(y_median_laptime_values))
#                x_median_values=list(range(1,laps+1))
                if str.lower(y_values_mode) =='lap':
                    return y_median_laptime_values
                else:
                    return y_median_deg_values
                
                "User entering driverlist='all' or empty str " 
                
        elif isinstance(driverlist,list): 
            for drivers in driverlist:
                LapTimesDf_SelectedDrivers=laptimes_df[laptimes_df['driver'].isin(driverlist)]
#                laps = LapTimesDf_SelectedDrivers['StintLaps'].max()
                y_median_laptime_values=[LapTimesDf_SelectedDrivers[sector][LapTimesDf_SelectedDrivers['StintLaps']==lap].median() for lap in laps_filter]
                y_median_deg_values=y_median_laptime_values-np.nanmin(np.array(y_median_laptime_values))
#                x_median_values=list(range(1,laps+1))
                if str.lower(y_values_mode) =='lap':
                    return y_median_laptime_values
                else:
                    return y_median_deg_values
                   
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
app=dash.Dash(__name__)
    
server=app.server
    
app.layout = html.Div(children=[
                      html.H2(id='tittle',children='Campos Engineering',style={'text-align':'center','fontSize':20}),
                                       # FIRST ROW DIVS with Selection Plot Dropdown, Event Input Box and Confirm Button
                      html.Div([                  
                              html.Div([
                                      dcc.Dropdown(id = 'plot options dropdown',
                                                   options=[{'label': i, 'value': i} for i in ['laptime','median','model']],
                                                   placeholder="Select Plot Calculations",
                                                   multi=True,
                                                   value=['laptime','median','model']
                                                   ),
                                        ],
                                                    style = dict(
                                                        width = '68%',
                                                        display = 'table-cell',
                                                        verticalAlign = "middle",
                                                        border="2px black solid"
                                                        ),
                                        ),
                            html.Div(['Introduce the Event Naming Convention to analyse:',
                                dcc.Input(id = 'event input', type = 'text', value = ''),
                                html.Button(
                                    children = 'Step 1: Confirm Event',
                                    id = 'event confirm button',
                                    type = 'submit',
                                    n_clicks = 0
                                    ),
                                ],
                                style = dict(
                                    width = '30%',
                                    display = 'table-cell',
                                    verticalAlign = "bottom",
                                                        border="2px black solid"
                                    )
                                ),
                                ],
                                style = dict(
                                    width = '100%',
                                    display = 'table',
                                    text_align = 'center',
                                                        border="2px black solid"
                                    )),
                                                    
                                      # SPACER
                            
                            html.P(), 
                            
                            # SECOND ROW DIVS with Selection Plot Dropdown, Event Input Box and Confirm Button
                            
                            html.Div([           
                            html.Div([                  
                                      dcc.Dropdown(id = 'compound options dropdown',
                                                   options=[{'label': i, 'value': i} for i in ['Prime','Option','Wet']],
                                                   placeholder="Select Compound",
                                                   value='Prime'),
                                        ],
                                                    style = dict(
                                                        width = '68%',
                                                        display = 'table-cell',
                                                        verticalAlign = "middle",
                                                        border="2px black solid"
                                                        ),
                                        ),
                        
                            html.Div(['Introduce the number of drivers [Top X drivers]:',
                                dcc.Input(id = 'top drivers input', type = 'text', value = 20),
                                html.Button(
                                    children = 'Step 2: Choose all the user inputs and  Click Here to refresh',
                                    id = 'refresh button',
                                    type = 'submit'
#                                    n_clicks = 0
                                    ),
                                ],
                                style = dict(
                                    width = '30%',
                                    display = 'table-cell',
                                    verticalAlign = "bottom",
                                    border="2px black solid",
                                    justify='center'
                                    
                                    )),
                                ],
                                style = dict(
                                    width = '100%',
                                    display = 'table',
                                    align = 'center',
                                    border="2px black solid"
                                    )),
                            
                                      # SPACER
                            
                            html.P(),
                            
                            html.P(), 
                            
                            # THIRD ROW DIVS with Selection of Mode (Deg or Laptimes) and Selection of S1/S2/S3/All
                            
                            html.Div([           
                            html.Div(['S1/S2/S3/Full Lap Mode',                 
                                      dcc.Dropdown(id = 'sector options dropdown',
                                                   options=[{'label': i, 'value': i} for i in ['S1','S2','S3','Full Lap']],
                                                   placeholder="Select Sector to analyse or full lap",
                                                   value='Full Lap')
                                        ],
                                                    style = dict(
                                                        width = '10%',
                                                        display = 'table-cell',
                                                        verticalAlign = "bottom",
                                                        border="2px black solid",
                                                        ),
                                        ),
                        
                            html.Div(['Deg/Laptimes Mode',
                                dcc.Dropdown(id = 'mode options dropdown',
                                                   options=[{'label': i, 'value': i[0:3]} for i in ['Degradation','Laptimes']],
                                                   placeholder="Select Graph Mode: Abs Deg or Laptimes",
                                                   value = 'Deg')
                                        ],
                                                    style = dict(
                                                        width = '10%',
                                                        display = 'table-cell',
                                                        verticalAlign = "bottom",
                                                        border="2px black solid"
                                                        ),
                                        ),
                            html.Div([
                                    'Select which laps to display on the plot. Median will be calculated only for these laps selected',
                                    dcc.Dropdown(
                                            id='laps filter',
                                            multi=True                                            
                                            )],
                                    style = dict(
                                                        width = '80%',
                                                        display = 'table-cell',
                                                        border="2px black solid"
                                                        )
                                            ),
                                ],
                                style = dict(
                                    width = '100%',
                                    display = 'table',
                                    align = 'center',
                                    border="2px black solid"
                                    )),
                            

                            
                                      # TRICK TO MAKE CALLBACKS WITHOUT AN OUTPUT
                            html.Div(id='hidden div', style=dict(display = 'none')),
                            
                                      # SPACER
                            
                            html.P(), 
                            
                                                    
                                    # GRAPH DIV ROW  
                                                    
                            html.Div([
                                dcc.Graph(id='feature-graphic')],
                                style={'display': 'inline-block', 'width': '100%'}),
                                    
                            html.P(),
                                     #EQ MODEL + COEFFS PBTAINED FROM MODEL FIT (50% Left)
                            html.Div([
                            html.Div([],style=dict(border="2px black solid",width='45%',display='table-cell')),
                                     
                                    #Other Option (50% Right)
                                             
                            html.Div([],style=dict(border="2px red solid",width='45%',display='table-cell'))
                            
                            ],style=dict(display='table',border="2px blue solid",width='100%'))
                            ])
                                                    
                                    
@app.callback(Output('hidden div','children'),
              [Input('event confirm button','n_clicks')],
              [State('event input','value')])

def event_starter(button_click,event_naming_convention):
    global User_Event,Heidi_DDBB,conn
    if button_click>0:
        Heidi_DDBB=DDBB()
        conn = Heidi_DDBB.db.connect()
        User_Event=Event(event_naming_convention,Heidi_DDBB)
        conn.close()
        Heidi_DDBB.db.dispose()

@app.callback([Output('laps filter','options'),
               Output('laps filter','value')],
              [Input('compound options dropdown','value'),
               Input('top drivers input','value')]
              )
def set_nr_of_lapfilter(val1,val2):
    
    if str.lower(comp) =='prime':
        maxlaps=User_Event.LapTimesDf_Prime['StintLaps'][User_Event.LapTimesDf_Prime['driver'].isin(User_Event.TopXDrivers(int(top_nr_dri)))].max()
        options=[dict(label=str(i),value=i) for i in range(1, maxlaps+1)]
        value=[d['value'] for d in options]
    elif str.lower(comp) =='option':
        maxlaps=User_Event.LapTimesDf_Option['StintLaps'][User_Event.LapTimesDf_Option['driver'].isin(User_Event.TopXDrivers(int(top_nr_dri)))].max()
        options=[dict(label=str(i),value=i) for i in range(1, maxlaps+1)]
        value=[d['value'] for d in options]

    return options,value        
     
@app.callback(Output('feature-graphic','figure'),
               [Input('plot options dropdown','value'),
                Input('compound options dropdown','value'),
                Input('top drivers input','value'),
                Input('sector options dropdown','value'),
                Input('mode options dropdown','value'),
                Input('refresh button','n_clicks'),
                Input('laps filter','value')
                
                ])

def update_graph(plot_options,compound,top_drivers,track_sector,y_values_mode,n_clicks,filtered_laps):
    
    global mode,sector, top_nr_dri,comp,laps
    
    mode=y_values_mode
    sector=track_sector
    top_nr_dri=top_drivers
    comp=compound
    laps=sorted(filtered_laps)
    DriverList=User_Event.TopXDrivers(int(top_nr_dri))
    
    
    y_median_deg=User_Event.GetLaptimesOrDegMedianByDriver(laps,compound,DriverList,mode,sector)
    try:
        TyreModelCoeffs,TyreModelCovar=User_Event.TyreModelCoeffs(laps,y_median_deg)
        TyreModelCoeffs=TyreModelCoeffs.tolist()
        y_model_deg=Eq_Model(np.array(laps),TyreModelCoeffs[0],TyreModelCoeffs[1],TyreModelCoeffs[2])
    except:
        pass
       
    if str.lower(track_sector) == 'full lap':
        sector="laptime_fuel_corrected"
    elif str.lower(track_sector) == 's1':
        sector = "s1"
    elif str.lower(track_sector) == 's2':
        sector = "s2"
    elif str.lower(track_sector) == 's3':
        sector = "s3"
    driverlist=DriverList
    if str.lower(compound) =='prime':
       
        laptimes_df=User_Event.LapTimesDf_Prime
        LapTimesDf_SelectedDrivers=laptimes_df[laptimes_df['driver'].isin(driverlist)]
        GroupByDriver=LapTimesDf_SelectedDrivers.groupby(['driver','Stint'])
        
    elif str.lower(compound) == 'option':
       
        laptimes_df=User_Event.LapTimesDf_Option
        LapTimesDf_SelectedDrivers=laptimes_df[laptimes_df['driver'].isin(driverlist)]
        GroupByDriver=LapTimesDf_SelectedDrivers.groupby(['driver','Stint'])
        
    else:
       
        laptimes_df=User_Event.LapTimesDf
        LapTimesDf_SelectedDrivers=laptimes_df[laptimes_df['driver'].isin(driverlist)]
        GroupByDriver=LapTimesDf_SelectedDrivers.groupby(['driver','Stint'])
    
    trace_drivers=[] #list initialization to plot data from drivers
    
    for drivers in driverlist:
            for stint in laptimes_df['Stint'][laptimes_df['driver']==drivers].unique().tolist():
                if str.lower(y_values_mode) == 'deg':
                    y_values=GroupByDriver.get_group((drivers,stint))[sector][GroupByDriver.get_group((drivers,stint))['StintLaps'].isin(laps)]-GroupByDriver.get_group((drivers,stint))[sector][GroupByDriver.get_group((drivers,stint))['StintLaps'].isin(laps)].min()
#                    x_values=GroupByDriver.get_group((drivers,stint))['StintLaps'][GroupByDriver.get_group(drivers)['StintLaps'].isin(laps)]
                    trace_drivers.append(go.Scatter(
                                                    x=laps,
                                                    y=y_values,
                                                    mode='lines+markers',
                                                    name=drivers + "_" + str(stint)
                
                
            ))
                else:
                    y_values=GroupByDriver.get_group((drivers,stint))[sector][GroupByDriver.get_group((drivers,stint))['StintLaps'].isin(laps)]
#                    x_values=GroupByDriver.get_group((drivers,stint))['StintLaps'][GroupByDriver.get_group((drivers,stint))['StintLaps'].isin(laps)]
                    trace_drivers.append(go.Scatter(
                                                    x=laps,
                                                    y=y_values,
                                                    mode='lines+markers',
                                                    name=drivers + "_" + str(stint)
                
                
            ))
                
            
                
                
                
    trace_median=go.Scatter(
            x=laps,
            y=y_median_deg,
            mode='lines+markers',
            name='Median of all drivers'
            
            
        )
    trace_model=go.Scatter(
            x=laps,
            y=y_model_deg,
            mode='lines+markers',
            name='Math Model Fit'            
            
        )
    
    layout=go.Layout(
            xaxis={'title': 'Laps'},
            yaxis={'title': 'Absolute Deg'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest'
        )
    data=[]
    if ('laptime' in plot_options) and ('median' in plot_options) and ('model' in plot_options):    
        trace_drivers.append(trace_median)
        trace_drivers.append(trace_model)
        data=trace_drivers
    elif ('laptime' in plot_options) and ('median' in plot_options):
        trace_drivers.append(trace_median)
        data=trace_drivers
    elif ('median' in plot_options) and ('model' in plot_options):
        trace_all=[]
        trace_all.append(trace_median)
        trace_all.append(trace_model)
        data = trace_all
    elif ('laptime' in plot_options) and ('model' in plot_options):
        trace_drivers.append(trace_model)
        data=trace_drivers
    elif ('laptime' in plot_options):
        data = trace_drivers
    elif ('median' in plot_options):
        data = [trace_median]
    elif ('model' in plot_options):
        data = [trace_model]
    conn.close()
    Heidi_DDBB.db.dispose()
    
    return dict(data=data,layout=layout)

#print(DriverList)
#print(Event.DriverList_Prime)
#print(Event.DriverList_Option)
#conn.close()
#Heidi_DDBB.db.dispose()

#####################MAIN PROGRAM##############################################                

if __name__ == "__main__":
    app.run_server(debug=False) 

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



#####compound csv to database
# STEP 0 db = create_engine('mysql://mf6bshg8uxot8src:nvd3akv0rndsmc6v@nt71li6axbkq1q6a.cbetxkdyhwsb.us-east-1.rds.amazonaws.com:3306/ss0isbty55bwe8te') CREATE DDBB Engine
# conn=db.connect()
# STEP 1 df_ddbb_compounds = pd.read_csv('TyreCompounds2019ImportDDBB')  
# STEP 2 df_ddbb_compounds=df_ddbb_compounds.replace(np.nan,'') Replace NaN by empty cells, if not ddbb crashes
# STEP 3 Option A) df_ddbb_compounds.to_sql('TyreAlloc', con=db, if_exists='replace') if something is wrong
# STEP 3 Option B) df_ddbb_compounds.to_sql('TyreAlloc', con=db, if_exists='append') if we want to just add a new event
# STEP 4 conn.close()
#        db.dispose()
###############################################################################