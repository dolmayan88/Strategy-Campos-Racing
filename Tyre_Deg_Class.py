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
import dash_table as dt
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output,State
import dash_daq as daq
import plotly.graph_objs as go
import plotly.figure_factory as ff

import Strategy

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

class Event():

    def __init__(self,Naming_convention='',livemargin=3600,pdftiming=False,category='',live=False):

        self.db = create_engine('mysql://mf6bshg8uxot8src:nvd3akv0rndsmc6v@nt71li6axbkq1q6a.cbetxkdyhwsb.us-east-1.rds.amazonaws.com:3306/ss0isbty55bwe8te')
        #One Single Access to DB
        self.CalendarDB=Event.getTotalTable("Calendar", None, None)
        # self.TyreAllocDB=self.getTotalTable("TyreAlloc", None, None)
        self.TyreModelsDB=self.getTotalTable("TyreModels", None, None)
        self.Name = str(Naming_convention)
        self.Chsip = self.Name.split("_")[0] if len(self.Name)>7 else ''
        self.Year = self.Name.split("_")[1][0:2] if len(self.Name)>7 else ''
        self.Track = self.Name.split("_")[1][5:8] if len(self.Name)>7 else ''
        self.Session = self.Name.split("_")[2] if len(self.Name)>7 else ''
        #One Single Access to DB
        self.CalendarDf=Event.getTotalTable("Calendar","Session_id",self.Name)
        self.TyreAllocDf=self.getTotalTable("TyreAlloc","Session",self.Name)
        
        if live:
            self.LapTimesDf=self.getdata(self.Name,livemargin,pdftiming,category) ######Check this is Miguel getdata
        else:
            if pdftiming:
                self.LapTimesDf=self.getTotalTable("PdfTiming","Session",self.Name)
            else:
                self.LapTimesDf=self.getTotalTable("RawTiming","Session",self.Name)

        self.PdfFlag = int(self.CalendarDf['pdf_flag'].values) if len(self.Name)>7 else ''
        self.TrackFuelPenalty = float(self.CalendarDf['fuel_penalty'].values) if len(self.Name)>7 else 1
        
        self.NrOfDifferentCompoundsUsed = len(np.unique(self.TyreAllocDf[['s1', 's2','s3','s4','s5','s6','s7','s8','s9','s10']].dropna().values))-1 if len(self.Name)>7 else 0
        self.PrimeCompound = "".join(c for c in str(self.CalendarDf['Prime_Tyre'].values) if c.isupper()) if len(self.Name)>7 else ''
        self.OptionCompound = "".join(c for c in str(self.CalendarDf['Option_Tyre'].values) if c.isupper()) if len(self.Name)>7 else ''
        self.DriverList = self.LapTimesDf.driver.unique().tolist() 
        self.NrofLaps = int(self.LapTimesDf.lap.max())
        # self.DriverStartPosition =self.TyreAllocDf[["driver","startpos"] if len(Naming_convention)>7 else ''
        self.DriverEndPosition = self.LapTimesDf[['driver','position']][self.LapTimesDf.lap==self.NrofLaps]
        self.DriverFinishLine= self.LapTimesDf['driver'][self.LapTimesDf.lap==self.NrofLaps]

        #Now this is valid live or not live
        
        self.LapTimesDf["laptime"]=self.LapTimesDf.laptime.replace('','0:00.000').apply(lambda x: convert2time(x))
        self.LapTimesDf["laptime_fuel_corrected"]=self.LapTimesDf.apply(lambda row: row.laptime + (row.lap-1)*self.TrackFuelPenalty,axis=1)
      
        if self.NrOfDifferentCompoundsUsed > 1:
            self.LapTimesDf['Tyre_Compound'] = self.LapTimesDf.apply (lambda row: self.label_compound(row), axis=1)
        else:
            self.LapTimesDf['Tyre_Compound'] = self.PrimeCompound
            
        self.LapTimesDf['pits']=self.LapTimesDf.apply(lambda row: self.label_pits(row),axis =1)
        self.LapTimesDf['Stint']=self.LapTimesDf['pits']+1
        self.LapTimesDf['StintLaps']=self.LapTimesDf.groupby(['driver','pits']).cumcount()+1
        self.LapTimesDf['laptime_gap_corrected'] = self.LapTimesDf.apply(lambda row: row['laptime_fuel_corrected'] - Strategy.Driver.traffic_loss_fun(row.interval), axis=1)
        self.LapTimesDf_Prime = self.LapTimesDf[self.LapTimesDf['Tyre_Compound']==self.PrimeCompound]
        self.MaxNrofLaps_Prime = self.LapTimesDf_Prime['StintLaps'].max()
        self.LapTimesDf_Option = self.LapTimesDf[self.LapTimesDf['Tyre_Compound']==self.OptionCompound]
        self.MaxNrofLaps_Option = self.LapTimesDf_Option['StintLaps'].max()
        self.DriverList_Prime = self.LapTimesDf['driver'][self.LapTimesDf['Tyre_Compound']==self.PrimeCompound].unique().tolist()
        self.DriverList_Option = self.LapTimesDf['driver'][self.LapTimesDf['Tyre_Compound']==self.OptionCompound].unique().tolist()
    @staticmethod
    def getPartialTable(DDBBColumn,DDBBTable,DDBBColumnFilter=None,DDBBValueFilter=None):

        db = create_engine('mysql://mf6bshg8uxot8src:nvd3akv0rndsmc6v@nt71li6axbkq1q6a.cbetxkdyhwsb.us-east-1.rds.amazonaws.com:3306/ss0isbty55bwe8te')
        conn=db.connect()
        if (DDBBColumnFilter==None) and (DDBBValueFilter==None):
            if isinstance(DDBBColumn,list):
            
                DDBB_df = pd.read_sql_query("SELECT " + ",".join(DDBBColumn) + " FROM `" + DDBBTable + "'",db)

            if (str.lower(DDBBColumn) == 'all') or (str.lower(DDBBColumn) == '*'):
                DDBB_df = pd.read_sql_query("SELECT * FROM `" + DDBBTable + "`",db)
                
            else:
                DDBB_df = pd.read_sql_query("SELECT " + DDBBColumn + " FROM `" + DDBBTable +"'",db)

            conn.close()
            db.dispose()
            if len(DDBB_df)>0:
                return DDBB_df
            else:
                print("No hay ningún dato en la BBDD perteneciente a esos filtros.")
        else:
            if isinstance(DDBBColumn,list):
            
                DDBB_df = pd.read_sql_query("SELECT " + ",".join(DDBBColumn) + " FROM `" + DDBBTable + "` WHERE `" + DDBBColumnFilter + "` LIKE '" + DDBBValueFilter +"'",db)

            if (str.lower(DDBBColumn) == 'all') or (str.lower(DDBBColumn) == '*'):
                DDBB_df = pd.read_sql_query("SELECT * FROM `" + DDBBTable + "` WHERE `" + DDBBColumnFilter + "` LIKE '" + DDBBValueFilter +"'",db)
                
            else:
                DDBB_df = pd.read_sql_query("SELECT " + DDBBColumn + " FROM `" + DDBBTable + "` WHERE `" + DDBBColumnFilter + "` LIKE '" + DDBBValueFilter +"'",db)
            conn.close()
            db.dispose()
            if len(DDBB_df)>0:
                return DDBB_df
            else:
                print("No hay ningún dato en la BBDD perteneciente a esos filtros.")
    @staticmethod
    def getTotalTable(DDBBTable,DDBBColumnFilter,DDBBValueFilter):
        db = create_engine('mysql://mf6bshg8uxot8src:nvd3akv0rndsmc6v@nt71li6axbkq1q6a.cbetxkdyhwsb.us-east-1.rds.amazonaws.com:3306/ss0isbty55bwe8te')
        conn=db.connect()
        if (DDBBColumnFilter==None) and (DDBBValueFilter==None):
            DDBB_df = pd.read_sql_query("SELECT * FROM `" + DDBBTable + "`",db)

        else:
            DDBB_df = pd.read_sql_query("SELECT * FROM `" + DDBBTable + "` WHERE `" + DDBBColumnFilter + "` LIKE '" + DDBBValueFilter +"'",db)

        conn.close()
        db.dispose()

        if len(DDBB_df)>0:
            return DDBB_df
        else:
            print("No hay ningún dato en la BBDD perteneciente a esos filtros.")

    def label_pits (self,row):

        if row['pits'] == 0 and row['InPit']==0:
           return int(0)
        if row['pits'] == 1 and row['InPit']==1:
           return int(0)
        if row['pits'] == 1 and row['InPit']==0:
           return int(1) 
        if row['pits'] == 2 and row['InPit']==1:
           return int(1)
        if row['pits'] == 2 and row['InPit']==0:
           return int(2)
        if row['pits'] == 3 and row['InPit']==1:
           return int(2)
        if row['pits'] == 3 and row['InPit']==0:
           return int(3) 
        if row['pits'] == 4 and row['InPit']==1:
           return int(3)
        if row['pits'] == 4 and row['InPit']==0:
           return int(4)
        if row['pits'] == 5 and row['InPit']==1:
           return int(4)
        if row['pits'] == 5 and row['InPit']==0:
           return int(5) 
        if row['pits'] == 6 and row['InPit']==1:
           return int(5)
        if row['pits'] == 6 and row['InPit']==0:
           return int(6)
        if row['pits'] == 7 and row['InPit']==1:
           return int(6)
        if row['pits'] == 7 and row['InPit']==0:
           return int(7)  
        if row['pits'] == 8 and row['InPit']==1:
           return int(7)
        if row['pits'] == 8 and row['InPit']==0:
           return int(8)
        if row['pits'] == 9 and row['InPit']==1:
           return int(8)
        if row['pits'] == 9 and row['InPit']==0:
           return int(9)          
        if row['pits'] == 10 and row['InPit']==1:
           return int(9)
        if row['pits'] == 10 and row['InPit']==0:
           return int(10)        
        
    def label_compound (self,row):
       for driver in self.DriverList:
           if row['driver'] == driver and row['pits'] == 0:
               return self.TyreAllocDf['s1'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]        
           if row['driver'] == driver and row['pits'] == 1:
               if row['InPit']== 1:                    
                   return self.TyreAllocDf['s1'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]                  
               else:
                   return self.TyreAllocDf['s2'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
              
           if row['driver'] == driver and row['pits'] == 2:
               if row['InPit']== 1:
                   
                   return self.TyreAllocDf['s2'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0] 
               else:
                   return self.TyreAllocDf['s3'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
           if row['driver'] == driver and row['pits'] == 3:
               if row['InPit']== 1:
                   return self.TyreAllocDf['s3'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.TyreAllocDf['s4'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
           if row['driver'] == driver and row['pits'] == 4:
               if row['InPit']== 1:
                   return self.TyreAllocDf['s4'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.TyreAllocDf['s5'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
           if row['driver'] == driver and row['pits'] == 5:
               if row['InPit']== 1:
                   return self.TyreAllocDf['s5'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.TyreAllocDf['s6'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
           if row['driver'] == driver and row['pits'] == 6:
               if row['InPit']== 1:
                   return self.TyreAllocDf['s6'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.TyreAllocDf['s7'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
           if row['driver'] == driver and row['pits'] == 7:
               if row['InPit']== 1:
                   return self.TyreAllocDf['s7'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.TyreAllocDf['s8'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
           
           if row['driver'] == driver and row['pits'] == 8:
               if row['InPit']== 1:
                   return self.TyreAllocDf['s8'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.TyreAllocDf['s9'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
           
           if row['driver'] == driver and row['pits'] == 9:
               if row['InPit']== 1:
                   return self.TyreAllocDf['s9'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
               else:
                   return self.TyreAllocDf['s10'][self.TyreAllocDf['driver']==row['driver']].values.tolist()[0]
                                  
    def TopXDrivers(self,top_number):
        if top_number != 'all':
            top_number = int(top_number)
            top_driver_list=self.DriverEndPosition['driver'][self.DriverEndPosition['position']<top_number+1].tolist()
        else:
            top_driver_list=self.LapTimesDf.driver.unique()
        return top_driver_list
                    
    def GetLaptimesOrDegMedianByDriver(self,laps_filter,compound,driverlist,y_values_mode,sector):
        global pace_tyre
        """
       Inputs:
           driverlist -> Here we can enter the filter for the drivers. 
                         It has to be a list of drivers or a string 'all' to take every driver into account.
       
           y_values_mode -> "deg" or "laptime" -> It is calculating the median of
                           all the laptimes or the median of the absolute deg,
                           that means median laptime per lap - minimum median of all the laps (All values greater than zero)
           
        """
        
        "User entering driverlist='all' or ''"

        if str.lower(compound) == 'prime':
            laptimes_df=User_Event.LapTimesDf_Prime
            #laptimes_df_cleanLaps=CleanLapTimesDf(laptimes_df)
        elif str.lower(compound) == 'option':
            laptimes_df=User_Event.LapTimesDf_Option
            #laptimes_df_cleanlaps = CleanLapTimesDf(laptimes_df)
        else:
            pass

        if (isinstance(driverlist,str)) and ((str.lower(driverlist) == 'all') or (driverlist == '')):              
            for drivers in self.DriverList:
                y_median_laptime_values=[laptimes_df[sector][laptimes_df['StintLaps']==lap].median() for lap in laps_filter]
                y_median_deg_values=y_median_laptime_values-np.nanmin(np.array(y_median_laptime_values))
#                x_median_values=list(range(1,laps+1))
                pace_tyre=min(y_median_laptime_values)
                if str.lower(y_values_mode) =='lap':
                    return y_median_laptime_values
                else:
                    return y_median_deg_values
                
                "User entering driverlist='all' or empty str " 
                
        elif isinstance(driverlist,list): 
            for drivers in driverlist:
                LapTimesDf_SelectedDrivers=laptimes_df[laptimes_df['driver'].isin(driverlist)]
#                laps = LapTimesDf_SelectedDrivers['StintLaps'].max()
                y_median_laptime_values=[LapTimesDf_SelectedDrivers[sector][(LapTimesDf_SelectedDrivers['StintLaps']==lap) & (LapTimesDf_SelectedDrivers['InPit']==0)].median() for lap in laps_filter]
                y_median_deg_values=y_median_laptime_values-np.nanmin(np.array(y_median_laptime_values))
                pace_tyre = min(y_median_laptime_values)
#                x_median_values=list(range(1,laps+1))
                if str.lower(y_values_mode) =='lap':
                    return y_median_laptime_values
                else:
                    return y_median_deg_values
                   
    def TyreModelCoeffs(self,laps,laptimes, linear = False):
        
#        global y_weight
        g=[0.25,0.25,0.25]
#        y_weight = np.empty(len(laptimes))
#        y_weight.fill(1)
#        y_weight[10:] = 0.1
        inf=np.inf
        if linear:
            bounds = ([-inf,0,-inf], [inf,1*10**-100,inf])
        else:
            bounds = (-np.inf, np.inf)
        coeff,cov = curve_fit(Eq_Model,laps,laptimes, bounds=bounds)#,sigma=y_weight,absolute_sigma=True)
        try:
            return coeff.tolist(),cov.tolist()
        except:
            return[0,0,0],[0,0,0]

######################DASH APP#################################################

User_Event = Event()
calendar = User_Event.CalendarDB
tyremodels = User_Event.TyreModelsDB
tyremodels['id'] = tyremodels.AI
calendar['session'] = calendar.apply(lambda row: row.session_id[-2], axis=1)

app=dash.Dash(__name__)

server=app.server
    
app.layout = html.Div(children=[
                      html.H2(id='tittle',children='Campos Engineering',style={'text-align':'center','fontSize':20}),
                                       # FIRST ROW DIVS with Selection Plot Dropdown, Event Input Box and Confirm Button
                      html.Div([                  
                              html.Div([
                                      dcc.Dropdown(id = 'plot options dropdown',
                                                   options=[{'label': i, 'value': i} for i in ['drivers','median','model']],
                                                   placeholder="Select Plot Calculations",
                                                   multi=True,
                                                   value=['drivers','median','model']
                                                   ),
                                        ],
                                                    style = dict(
                                                        width = '68%',
                                                        display = 'table-cell',
                                                        verticalAlign = "middle"
                                                        ),
                                        ),
                            html.Div(['Introduce the Event Naming Convention to analyse:',
                                # dcc.Input(id = 'event input', type = 'text', value = ''),
                                dcc.Dropdown(id = 'event input',
                                             options = [{'label': i, 'value': i} for i in calendar[(calendar.pdf_flag=='1') & (calendar.session =='R')].session_id],
                                             multi = False,
                                             value = ''),
                                # html.Button(
                                #     children = 'Step 1: Confirm Event',
                                #     id = 'event confirm button',
                                #     type = 'submit',
                                #     n_clicks = 0
                                #     ),
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
                                                        
                                    )),
                                                    
                                      # SPACER
                            
                            html.P(), 
                            
                            # SECOND ROW DIVS with Selection Plot Dropdown, Event Input Box and Confirm Button
                            
                            html.Div([           
                            html.Div([                  
                                      dcc.Dropdown(id = 'compound options dropdown',
                                                   options=[{'label': i, 'value': i} for i in ['Prime','Option','Wet']],
                                                   placeholder="Select Compound"),
                                        ],
                                                    style = dict(
                                                        width = '68%',
                                                        display = 'table-cell',
                                                        verticalAlign = "middle"
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
                                    align = 'center'
                                    )),
                            
                                      # SPACER
                            
                            html.P(),
                            
                            html.P(), 
                            
                            # THIRD ROW DIVS with Selection of Mode (Deg or Laptimes) and Selection of S1/S2/S3/All
                            
                            html.Div([           
                            html.Div(['S1/S2/S3/Full Lap Mode',                 
                                      dcc.Dropdown(id = 'sector options dropdown',
                                                   options=[{'label': i, 'value': i} for i in ['S1','S2','S3','Full Lap','full lap - gap corrected']],
                                                   placeholder="Select Sector to analyse or full lap",
                                                   value='Full Lap')
                                        ],
                                                    style = dict(
                                                        width = '10%',
                                                        display = 'table-cell',
                                                        verticalAlign = "bottom",
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
                                                        verticalAlign = "bottom"
                                                        ),
                                        ),
                            html.Div([
                                    'Select which laps to display on the plot. Median will be calculated only '
                                    'for these laps selected',
                                    dcc.Dropdown(
                                            id='laps filter',
                                            multi=True
                                            )],
                                    style = dict(
                                                        width = '80%',
                                                        display = 'table-cell'
                                                        )
                                            ),
                                ],
                                style = dict(
                                    width = '100%',
                                    display = 'table',
                                    align = 'center'
                                    )),
                            

                            
                                      # TRICK TO MAKE CALLBACKS WITHOUT AN OUTPUT
                            html.Div(id='hidden div', style=dict(display = 'none')),

                            html.Div(id='hidden div 2', style=dict(display = 'none')),
                            
                                      # SPACER
                            
                            html.P(),

                            html.Div([
                                    'Select which drivers to display on the plot. Median will be calculated only for these laps selected',
                                    dcc.Dropdown(
                                            id='drivers filter',
                                            multi=True
                                            )],
                                    style = dict(
                                                        width = '100%',
                                                        display = 'table-cell'
                                                        )
                                            ),

                            html.P(),

                            # GRAPH DIV ROW
                                                    
                            html.Div([
                                dcc.Graph(id='feature-graphic')],
                                style={'display': 'inline-block', 'width': '100%'}),
                                    
                            html.P(),
                                     #EQ MODEL + COEFFS PBTAINED FROM MODEL FIT (50% Left)
                            html.Div([
                            html.Div(id="graph-container", children=[dcc.Graph(id='coeffs table')
                                
                                    ])
                                #,
                                     
                                    #Other Option (50% Right)
                                             
                            #html.Div([],style=dict(width='45%',display='table-cell'))
                            
                            ],style=dict(display='table',width='100%')),

                            html.Div([html.Div(daq.BooleanSwitch(id='Linear_model_switch',
                                                                 label='Linear Model',
                                                                 labelPosition = 'left',
                                                                 color='green',
                                                                 on=False),
                                               style={'display':'table-cell',
                                                      'width': '10%'}),
                                      html.Button(children='Send Model to DDBB',
                                                  id='DDBB Model button',
                                                  type='submit',
                                                  style={'display': 'table-cell',
                                                         'width' : '20%'}),
                                      html.Button(children='Update Table',
                                                  id='update_table_button',
                                                  type='submit',
                                                  style={'display': 'table-cell',
                                                         'width' : '20%'}),
                                      ],
                                     style={'display':'table',
                                            'width': '100%'}),
    html.Div([html.Div(dt.DataTable(id='database_table',
                                    columns=[{'name': i, 'id': i, 'deletable': False} for i in tyremodels.columns if i not in ['id', 'AI', 'Filtered Laps']],
                                    data = tyremodels.to_dict('records'),
                                    editable = False,
                                    filter_action = "native",
                                    sort_action = "native",
                                    sort_mode = 'multi',
                                    row_selectable = 'multi',
                                    row_deletable = False,
                                    selected_rows = [],
                                    page_action = 'native',
                                    page_current = 0,
                                    page_size = 10,
                                    style_data={'whiteSpace': 'normal',
                                                'height': 'auto',
                                                'lineHeight': '15px'
                                                },
                                    style_table={'overflowX': 'auto'},
                                    style_cell={
                                        # all three widths are needed
                                        'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
                                        'overflow': 'hidden',
                                        'textOverflow': 'ellipsis',
                                    }
                                    ),
                       style={'display': 'table-cell',
                              'width': '50%',
                              'padding': 60}),
              html.Div(id='Timeloss_chart',
                       style={'padding': -40})],
             style={'display': 'table'}),
    dcc.Input(id='filter-query-input', placeholder='Enter filter query', style={'width': '100%'})
                            ])

@app.callback(
    Output('database_table', 'filter_query'),
    [Input('filter-query-input', 'value')]
)
def write_query(query):
    print(query)
    if query is None:
        return ''
    return query

@app.callback(Output('Timeloss_chart', 'children'),
              [Input('database_table', 'derived_virtual_row_ids'),
               Input('database_table', 'selected_row_ids'),
               Input('database_table', 'active_cell')]
              )
def plot_tyremodels(row_ids, selected_row_ids, active_cell):
    # global tyremodels
    selected_id_set = set(selected_row_ids or [])

    my_df = tyremodels
    if row_ids is None:
        # pandas Series works enough like a list for this to be OK
        row_ids = tyremodels['id']
    else:
        my_df = tyremodels[tyremodels.id.isin(row_ids)]

    active_row_id = active_cell['id'] if active_cell else None

    # colors = ['#FF69B4' if id == active_row_id
    #           else '#7FDBFF' if id in selected_id_set
    #           else '#0074D9'
    #           for id in row_ids]
    my_df['tyre'] = my_df.apply(lambda row: Strategy.Tyre(row.A, row.B, row.C), axis=1)
    x=list(range(0,30))
    figure = go.Figure()
    for id in my_df.id.unique():
        width=1
        if selected_row_ids is not None:
            if id in selected_row_ids:
                width=3
        name = str(my_df[my_df.id==id][['Session', 'Compound']].values[0])
        if len(str(my_df[my_df.id == id]['TopDriversX'].values[0])) == 3:
            name += ' ' + str(my_df[my_df.id == id]['TopDriversX'].values[0])
        figure.add_trace(go.Scatter(x=x,
                                    y=list(my_df[my_df.id==id].tyre.values[0].loss_fun(max(x)+1).values()),
                                    name=name,
                                    line={'width': width}
                                    )
                         )
    figure.update_layout({'yaxis': {'title': 'time loss [s]',
                                    'range': [0,10]}})
    return [
        dcc.Graph(
            id='tyreloss',
            figure=figure,
        )
    ]


@app.callback(Output('hidden div','children'),
              # [Input('event confirm button','n_clicks')],
              [Input('event input','value')])
def eventclasscreation(event_naming_convention):
    global User_Event,conn
    # if button_click>0:
    User_Event=Event(Naming_convention=event_naming_convention,pdftiming=True)
        # conn = User_Event.db.connect()
        # conn.close()
        # User_Event.db.dispose()


@app.callback([Output('laps filter','options'),
               Output('laps filter','value'),
               Output('drivers filter','options'),
               Output('drivers filter','value')],
              [Input('compound options dropdown','value'),
               Input('top drivers input','value'),]
              )
def updatelapfilter(comp,top_nr_dri):

    driveroptions = [dict(label=str(i), value=str(i)) for i in User_Event.TopXDrivers(top_nr_dri)]
    drivervalue = [d['value'] for d in driveroptions]

    if str.lower(comp) =='prime':

        maxlaps=User_Event.LapTimesDf_Prime['StintLaps'][User_Event.LapTimesDf_Prime['driver'].isin(drivervalue)].max()
        print('entrando en max laps!!!')
        print(maxlaps)
        lapsoptions=[dict(label=str(i),value=i) for i in range(1, maxlaps)]
        lapsvalue=[d['value'] for d in lapsoptions]

    elif str.lower(comp) =='option':

        maxlaps=User_Event.LapTimesDf_Option['StintLaps'][User_Event.LapTimesDf_Option['driver'].isin(drivervalue)].max()
        lapsoptions=[dict(label=str(i),value=i) for i in range(1, maxlaps)]
        lapsvalue=[d['value'] for d in lapsoptions]

    return lapsoptions,lapsvalue,driveroptions,drivervalue

@app.callback(Output('feature-graphic','figure'),
               [Input('plot options dropdown','value'),
                Input('compound options dropdown','value'),
                Input('drivers filter','value'),
                Input('sector options dropdown','value'),
                Input('mode options dropdown','value'),
                Input('refresh button','n_clicks'),
                Input('laps filter','value'),
                Input('Linear_model_switch','on')
                ])

def updategraph(plot_options,compound,driverfilterlist,track_sector,y_values_mode,n_clicks,filtered_laps, linear_model):
    
    global laps, TyreModelCoeffs, actual_comp, Driverlist #Global Variables to update Table

    laps=sorted(filtered_laps)
    sector = GetSectorMode(track_sector)
    #Driverlist=User_Event.TopXDrivers(int(top_drivers))
    Driverlist = driverfilterlist
    #Get median values
    y_median_deg=User_Event.GetLaptimesOrDegMedianByDriver(laps,compound,Driverlist,y_values_mode,sector)
    print(y_median_deg)
    print(len(y_median_deg))

    #Get Model Values
    TyreModelCoeffs,TyreModelCovar=User_Event.TyreModelCoeffs(laps,y_median_deg,linear_model)
    y_model_deg=Eq_Model(np.array(laps),TyreModelCoeffs[0],TyreModelCoeffs[1],TyreModelCoeffs[2])

    #Get Compound Selected Data Values for getting driver values
    actual_comp, laptimes_df, GroupByDriver = GetCompoundData(Driverlist, compound)


    #Definition of Scatters to plot (Drivers + Median + Math Model)
    trace_drivers = GetDriversData(GroupByDriver, Driverlist, laps, laptimes_df, sector, y_values_mode)

    trace_median=go.Scatter(
            x=laps,
            y=y_median_deg,
            mode='lines+markers',
            line=dict(color='red', width=8, dash='dash'),
            name='Median')

    trace_model=go.Scatter(
            x=laps,
            y=y_model_deg,
            mode='lines+markers',
            line=dict(color='black', width=8, dash='dot'),
            name='Tyre Model Fit')
    
    layout=go.Layout(
            xaxis={'title': 'Laps'},
            yaxis={'title': 'Absolute Deg'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 0},
            hovermode='closest')

    data=[]
    if ('drivers' in plot_options) and ('median' in plot_options) and ('model' in plot_options):    
        trace_drivers.append(trace_median)
        trace_drivers.append(trace_model)
        data=trace_drivers
    elif ('drivers' in plot_options) and ('median' in plot_options):
        trace_drivers.append(trace_median)
        data=trace_drivers
    elif ('median' in plot_options) and ('model' in plot_options):
        trace_all=[]
        trace_all.append(trace_median)
        trace_all.append(trace_model)
        data = trace_all
    elif ('drivers' in plot_options) and ('model' in plot_options):
        trace_drivers.append(trace_model)
        data=trace_drivers
    elif ('drivers' in plot_options):
        data = trace_drivers
    elif ('median' in plot_options):
        data = [trace_median]
    elif ('model' in plot_options):
        data = [trace_model]

    # conn.close()
    # User_Event.db.dispose()
    
    return dict(data=data,layout=layout)#,new_table_figure


def GetCompoundData(Driverlist, compound):
    global actual_comp
    if str.lower(compound) == 'prime':
        actual_comp = User_Event.PrimeCompound
        laptimes_df = User_Event.LapTimesDf_Prime
        LapTimesDf_SelectedDrivers = laptimes_df[laptimes_df['driver'].isin(Driverlist)]
        GroupByDriver = LapTimesDf_SelectedDrivers.groupby(['driver', 'Stint'])

    elif str.lower(compound) == 'option':
        actual_comp = User_Event.OptionCompound
        laptimes_df = User_Event.LapTimesDf_Option
        LapTimesDf_SelectedDrivers = laptimes_df[laptimes_df['driver'].isin(Driverlist)]
        GroupByDriver = LapTimesDf_SelectedDrivers.groupby(['driver', 'Stint'])

    else:
        actual_comp = User_Event.PrimeCompound
        laptimes_df = User_Event.LapTimesDf
        LapTimesDf_SelectedDrivers = laptimes_df[laptimes_df['driver'].isin(Driverlist)]
        GroupByDriver = LapTimesDf_SelectedDrivers.groupby(['driver', 'Stint'])
    return actual_comp, laptimes_df, GroupByDriver


def GetSectorMode(track_sector):

    if str.lower(track_sector) == 'full lap':
        sector = "laptime_fuel_corrected"
    elif str.lower(track_sector) == 'full lap - gap corrected':
        sector = "laptime_gap_corrected"
    elif str.lower(track_sector) == 's1':
        sector = "s1"
    elif str.lower(track_sector) == 's2':
        sector = "s2"
    elif str.lower(track_sector) == 's3':
        sector = "s3"
    return sector


def CleanLapTimesDf(laptimes_df):
    pass


    #return laptimes_df_cleanlaps

def GetDriversData(GroupByDriver, driverlist, laps, laptimes_df, sector, y_values_mode):
    trace_drivers = []  # list initialization to plot data from drivers
    for drivers in driverlist:
        for stint in laptimes_df['Stint'][laptimes_df['driver'] == drivers].unique().tolist():
            if str.lower(y_values_mode) == 'deg':
                y_values = GroupByDriver.get_group((drivers, stint))[sector][:-1][
                               GroupByDriver.get_group((drivers, stint))['StintLaps'].isin(laps)] - \
                           GroupByDriver.get_group((drivers, stint))[sector][:-1][
                               GroupByDriver.get_group((drivers, stint))['StintLaps'].isin(laps)].min()
                #                    x_values=GroupByDriver.get_group((drivers,stint))['StintLaps'][GroupByDriver.get_group(drivers)['StintLaps'].isin(laps)]
                trace_drivers.append(go.Scatter(
                    x=laps,
                    y=y_values,
                    mode='lines+markers',
                    name=drivers + "_" + str(stint)))
            else:
                y_values = GroupByDriver.get_group((drivers, stint))[sector][:-1][
                    GroupByDriver.get_group((drivers, stint))['StintLaps'].isin(laps)]
                #                    x_values=GroupByDriver.get_group((drivers,stint))['StintLaps'][GroupByDriver.get_group((drivers,stint))['StintLaps'].isin(laps)]
                trace_drivers.append(go.Scatter(
                    x=laps,
                    y=y_values,
                    mode='lines+markers',
                    name=drivers + "_" + str(stint)))

    return trace_drivers

@app.callback(Output('coeffs table', 'figure'), [Input('feature-graphic','figure'),
                                                 Input('compound options dropdown','value'),
                                                 Input('top drivers input','value'),
                                                 Input('sector options dropdown', 'value')])
def updatetabledata(figure,comp,top_nr_dri, mode):
    global dff
    dff = pd.DataFrame(data=[{'A': round(TyreModelCoeffs[0], 3), 'B': round(TyreModelCoeffs[1], 3),
                              'C': round(TyreModelCoeffs[2], 3), 'TyrePace': pace_tyre,'Session': str(User_Event.Name),
                              'PO': comp, 'Compound': actual_comp, 'TopDriversX':",".join(
                                  [str(items) for items in Driverlist]),
                              'Filtered Laps': ",".join(
                                  [str(items) for items in laps]),
                              'Mode': str(mode)}])  # replace with your own data processing code
    new_table_figure = ff.create_table(dff)
    return new_table_figure

@app.callback(Output('graph-container', 'style'), [Input('compound options dropdown','value')])
def hide_graph(my_input):
    if my_input:
        return dict(width='100%',display='block')
    return dict(display='none')

@app.callback(Output('hidden div 2', 'children'), [Input('DDBB Model button','n_clicks')])
def send_model_to_DDBB(my_input):
    conn = User_Event.db.connect()

    dff.to_sql('TyreModels', con=conn, if_exists='append',index=False)
    conn.close()
    User_Event.db.dispose()
    return dict(display='none')

@app.callback([Output('database_table', 'data'),
               Output('database_table', 'filter_query')],
              [Input('update_table_button', 'n_clicks')])
def update_table_database(nclicks):
    global tyremodels
    User_Event = Event()
    tyremodels = User_Event.TyreModelsDB
    tyremodels['id'] = tyremodels.AI
    return tyremodels.to_dict('records'), ''

#####################MAIN PROGRAM##############################################                

if __name__ == "__main__":

    app.run_server(debug=False)

# event=Event("F2_19R01BAH_R1",pdftiming=True)

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
# STEP 1 df_ddbb_compounds = pd.read_csv('TyreCompounds2019ImportDDBB.csv')  
# STEP 2 df_ddbb_compounds=df_ddbb_compounds.replace(np.nan,'') Replace NaN by empty cells, if not ddbb crashes
# STEP 3 Option A) df_ddbb_compounds.to_sql('TyreAlloc', con=db, if_exists='replace') if something is wrong
# STEP 3 Option B) df_ddbb_compounds.to_sql('TyreAlloc', con=db, if_exists='append') if we want to just add a new event
# STEP 4 conn.close()
#        db.dispose()
###############################################################################