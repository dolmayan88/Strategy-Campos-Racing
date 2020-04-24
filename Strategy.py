# import math
# import json
# import time
# import datetime
import numpy as np
import pandas
import itertools
# import dash
# import dash_html_components as html
# import dash_core_components as dcc
# import dash_daq as daq
# from dash.dependencies import Input, Output, State
# import dash_auth
# from flask_caching import Cache
# import plotly.graph_objs as go
# from sqlalchemy import create_engine
# import os


class Tyre:

    def __init__(self, a, b, c, delta_pace=0, name=''):
        self.name = name
        self.delta_pace = delta_pace
        self.a = a
        self.b = b
        self.c = c

    def eq_model(self, lap, a, b, c):
        return a*lap+b*np.exp(c*lap) + self.delta_pace

    def lapsdf(self, lapsinstint):
        if type(lapsinstint) == int:
            lapsinstint = [lapsinstint]
        stintlap = []
        for lapsstint in lapsinstint:
            stintlap = stintlap + list(range(1, lapsstint + 1))
        return pandas.DataFrame(data={'Lap': list(range(1, len(stintlap) + 1)),
                                      'StintLap': stintlap})

    def lossdf(self, lapsinstint):
        df = self.lapsdf(lapsinstint)
        df['TyreLoss'] = df.apply(lambda lap: self.eq_model(lap['StintLap'], self.a, self.b, self.c), axis=1)
        return df

    def loss_fun(self, lapsinstint):
        return {lap: self.eq_model(lap, self.a, self.b, self.c) for lap in range(1, lapsinstint + 1)}


class Driver:

    def __init__(self, delta_pace=0, name=''):
        self.delta_pace = delta_pace
        self.name = name

    def lapsdf(self, lapsinstint):
        if type(lapsinstint) == int:
            lapsinstint = [lapsinstint]
        stintlap = []
        for lapsstint in lapsinstint:
            stintlap = stintlap + list(range(1, lapsstint + 1))
        return pandas.DataFrame(data={'Lap': list(range(1, len(stintlap) + 1)),
                                      'StintLap': stintlap})

    def lossdf(self, lapsinstint):
        df = self.lapsdf(lapsinstint)
        df['DriverLoss'] = df.apply(lambda lap: self.delta_pace, axis=1)
        return df

    def loss_fun(self, lapsinstint):
        return {lap: self.delta_pace for lap in range(1, lapsinstint + 1)}


class Stint:

    def __init__(self, laps, tyre, driver, refpace, initialloss):
        self.laps = laps
        self.tyre = tyre
        self.driver = driver
        self.refpace = refpace
        self.initialloss = initialloss
        self.laptimes = self.laptimes_fun()

    def laptimes_fun(self):
        tyreloss = self.tyre.loss_fun(self.laps)
        driverloss = self.driver.loss_fun(self.laps)
        laptimes = {lap: tyreloss[lap] + driverloss[lap] + self.refpace for lap in range(1, self.laps + 1)}
        laptimes[1] = laptimes[1] + self.initialloss
        return laptimes


class Event:

    def __init__(self, name=''):
        self.name = name
        self.refpace = 100
        self.pitloss = 30
        self.pitloss_vsc = 25
        self.pitloss_sc = 20
        self.gridloss = 0.1
        self.startingposition = {'': 1}


class DriverStrategy:

    def __init__(self, lapslist, tyrelist, driver, event, name='', sclist=[], vsclist=[]):
        self.name = name
        self.lapslist = lapslist
        self.tyrelist = tyrelist
        self.driver = driver
        self.refpace = event.refpace
        self.pitloss = event.pitloss
        self.initialloss = event.gridloss*event.startingposition[driver.name]
        self.sclist = sclist
        self.vsclist = vsclist
        self.stints = self.createstints()
        self.laptimes = self.laptimes_fun()
        self.data = self.data_fun()
        self.racetime = self.data['cumulative_time'].max()
        self.totallaps = self.data['lap'].max()
        self.avgtime = self.racetime/self.totallaps

    def createstints(self):
        stintnumber = 0
        stints = {}
        for stintlaps in self.lapslist:
            stintnumber = stintnumber + 1
            if stintnumber == 1:
                initloss = self.initialloss
            else:
                initloss = self.pitloss
            stintinstance = Stint(stintlaps, self.tyrelist[stintnumber-1], self.driver, self.refpace, initloss)
            stints[stintnumber] = stintinstance
        return stints

    def laptimes_fun(self):
        laptimes_list = []
        for stint in self.stints.values():
            laptimes_list = laptimes_list + list(stint.laptimes.values())
        lap = list(range(1, len(laptimes_list) + 1))
        return dict(zip(lap, laptimes_list))

    def data_fun(self):
        data = pandas.DataFrame(data=list(self.laptimes.keys()), columns=['lap'])
        data['laptimes'] = list(self.laptimes.values())
        stintlaps_list = []
        stint_counter = 0
        stint_list = []
        tyre_list = []
        for stint in self.stints.values():
            stint_counter = stint_counter + 1
            stintlaps_list = stintlaps_list + list(range(1, stint.laps + 1))
            stint_list = stint_list + [stint_counter] * stint.laps
            tyre_list = tyre_list + [self.tyrelist[stint_counter - 1]] * stint.laps
        data['stintlap'] = stintlaps_list
        data['stintn'] = stint_list
        data['tyre'] = tyre_list
        data['driver'] = [self.driver] * len(self.laptimes)
        data['cumulative_time'] = data['laptimes'].cumsum()
        return data


class Strategy:

    def __init__(self, driver_strategylist):
        self.driver_strategylist = driver_strategylist
        self.racetimes = dict(zip([driver.driver for driver in driver_strategylist],
                                  [driver.racetime for driver in driver_strategylist]))
        self.avgtimes = dict(zip([driver.driver for driver in driver_strategylist],
                                 [driver.avgtime for driver in driver_strategylist]))
        self.winner = min(self.racetimes, key=self.racetimes.get)
        self.quickestavg = min(self.avgtimes.values())
        self.undercut = self.undercut_fun(20)
        self.compounddeltatime = self.compounddeltatime_fun(20)
        self.raceplot = self.raceplot_fun()

    def raceplot_fun(self, overtaking=False):
        raceplot = pandas.DataFrame()
        if not overtaking:
            for driver_strategy in self.driver_strategylist:
                raceplot[driver_strategy.name] = \
                    driver_strategy.data['lap'] * self.quickestavg - driver_strategy.data['cumulative_time']
        return raceplot

    def undercut_fun(self, laps):
        tyre_list = []
        undercut = pandas.DataFrame()
        for driver_strategy in self.driver_strategylist:
            tyre_list = tyre_list + driver_strategy.tyrelist
        tyre_list = set(tyre_list)
        if len(tyre_list) > 1:
            tyre_combinations = list(itertools.combinations(tyre_list, 2))
            for tyre_pair in tyre_combinations:
                undercut[tyre_pair[0].name + ' to ' + tyre_pair[1].name] = [tyre_pair[1].delta_pace - x for x in
                                                                            list(tyre_pair[0].loss_fun(laps).values())]
                undercut[tyre_pair[1].name + ' to ' + tyre_pair[0].name] = [tyre_pair[0].delta_pace - x for x in
                                                                            list(tyre_pair[1].loss_fun(laps).values())]
        return undercut

    def compounddeltatime_fun(self, laps):
        tyre_list = []
        compoundloss = pandas.DataFrame()
        compounddeltatime = pandas.DataFrame()
        for driver_strategy in self.driver_strategylist:
            tyre_list = tyre_list + driver_strategy.tyrelist
        tyre_list = set(tyre_list)
        if len(tyre_list) > 1:
            tyre_combinations = list(itertools.combinations(tyre_list, 2))
            for tyre in tyre_list:
                compoundloss[tyre.name] = list(tyre.loss_fun(laps).values())
                compoundloss[tyre.name] = compoundloss[tyre.name].cumsum()
            for tyre_pair in tyre_combinations:
                compounddeltatime[tyre_pair[0].name + ' vs ' + tyre_pair[1].name] = \
                    compoundloss[tyre_pair[1].name] - compoundloss[tyre_pair[0].name]
        return compounddeltatime
