<<<<<<< HEAD
# import math
# import json
# import time
# import datetime
import numpy as np
import pandas
import itertools
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq
from dash.dependencies import Input, Output, State
import dash_auth
# from flask_caching import Cache
# import plotly.graph_objs as go
from sqlalchemy import create_engine
# import os


def listintosublist(primarylist, sublist):
    newlist = []
    for lensublist in sublist:
        newsublist = []
        for i in range(0, lensublist):
            newsublist.append(primarylist.pop(0))
        newlist.append(newsublist)
    return newlist


class Tyre:

    def __init__(self, a, b, c, delta_pace=0, name=''):
        self.name = name
        self.delta_pace = delta_pace
        self.a = a
        self.b = b
        self.c = c

    def eq_model(self, lap, a, b, c):
        return a*lap+b*np.exp(c*lap) + self.delta_pace

    def loss_fun(self, lapsinstint):
        return {lap: self.eq_model(lap, self.a, self.b, self.c) for lap in range(1, lapsinstint + 1)}


class Driver:

    def __init__(self, delta_pace=0, name=''):
        self.delta_pace = delta_pace
        self.name = name
        self.max_time_loss = 0.5
        self.starting_time_loss = 0
        self.gap_at_timeloss0 = 1

    def loss_fun(self, lapsinstint):
        return {lap: self.delta_pace for lap in range(1, lapsinstint + 1)}

    def traffic_loss_fun(self, gap):
        if 0 < gap < self.gap_at_timeloss0:
            return gap / self.gap_at_timeloss0 * (self.starting_time_loss - self.max_time_loss) + self.max_time_loss
        else:
            return 0


class Stint:

    def __init__(self, laps, tyre, driver, refpace, initialloss, gaps):
        self.laps = laps
        self.tyre = tyre
        self.driver = driver
        self.refpace = refpace
        self.initialloss = initialloss
        self.gaps = gaps
        self.laptimes = self.laptimes_fun()

    def laptimes_fun(self):
        tyreloss = self.tyre.loss_fun(self.laps)
        driverloss = self.driver.loss_fun(self.laps)
        laptimes = {lap: tyreloss[lap] + driverloss[lap] + self.refpace + self.driver.traffic_loss_fun(self.gaps[lap-1])
                    for lap in range(1, self.laps + 1)}
        laptimes[1] = laptimes[1] + self.initialloss
        return laptimes


class Event:

    def __init__(self, name=''):
        self.name = name
        self.laps = 30
        self.refpace = 100
        self.pitloss = 30
        self.pitloss_vsc = 25
        self.pitloss_sc = 20
        self.gridloss = 0.1
        self.startingposition = 1  # {'': 1}


class DriverStrategy:

    def __init__(self, lapslist, tyrelist, driver, event, name='', gaps=False):
        self.name = name
        self.lapslist = lapslist
        self.tyrelist = tyrelist
        self.driver = driver
        self.refpace = event.refpace
        self.pitloss = event.pitloss
        self.initialloss = event.gridloss*event.startingposition  # [driver.name]
        if gaps:
            self.gaps = gaps
        else:
            self.gaps = [[2]*laps for laps in lapslist]
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
            stintinstance = Stint(stintlaps, self.tyrelist[stintnumber-1], self.driver, self.refpace, initloss,
                                  self.gaps[stintnumber-1])
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
        self.laptimes_df = self.laptimes_df_fun()
        self.gaps_df = self.gaps_df_fun()
        self.positions_df = self.laptimes_df.cumsum().rank(axis=1, method='dense')
        self.gapsahead_df = self.gapsahead_df_fun()

    def raceplot_fun(self):
        raceplot = pandas.DataFrame()
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

    def laptimes_df_fun(self):
        laptimes_df = pandas.DataFrame()
        for driver_strategy in self.driver_strategylist:
            laptimes_df[driver_strategy.name] = pandas.Series(driver_strategy.laptimes,
                                                              index=driver_strategy.laptimes.keys())
        return laptimes_df

    def gaps_df_fun(self):
        cumulative_df = self.laptimes_df.cumsum()
        gaps_df = cumulative_df.apply(lambda row: row-cumulative_df.min(axis=1))
        return gaps_df

    def gapsahead_df_fun(self):
        racelaps = len(self.laptimes_df.index)
        data = []
        for lap in range(1, racelaps + 1):
            data.append(list(self.laptimes_df.cumsum().sort_values(lap, axis=1).columns))
        data = np.array(data)
        driver_byposition = pandas.DataFrame(data=data,
                                             columns=range(1, len(self.laptimes_df.columns) + 1),
                                             index=range(1, racelaps + 1))
        gapsahead_df = pandas.DataFrame()
        for strategy in self.gaps_df.columns:
            temp = []
            for lap in self.gaps_df.index:
                if self.gaps_df[strategy][lap] == 0:
                    temp.append(0)
                else:
                    myposition = self.positions_df[strategy][lap]
                    mycompetitor = driver_byposition[myposition-1][lap]
                    temp.append(self.gaps_df[strategy][lap]-self.gaps_df[mycompetitor][lap])
            gapsahead_df[strategy] = temp
        return gapsahead_df


class Race:

    def __init__(self, event, strategylist, maxiterations, tolerance):
        self.event = event
        self.maxiterations = maxiterations
        self.tolerance = tolerance
        self.strategies, self.strategy = self.calcrace_fun(strategylist)

    def calcrace_fun(self, strategylist):
        delta = 10 ^ 1000
        i = 0
        gaps = {driverstrategy.name: [[2]*laps for laps in driverstrategy.lapslist] for driverstrategy in strategylist}
        strategies = {driverstrategy.name: driverstrategy for driverstrategy in strategylist}
        nameslist = list(strategies.keys())
        strategy = {}
        while delta > self.tolerance or i < self.maxiterations:
            i = i + 1
            prevgaps = gaps
            strategies = {name: DriverStrategy(strategies[name].lapslist,
                                               strategies[name].tyrelist,
                                               strategies[name].driver,
                                               self.event,
                                               name=name,
                                               gaps=gaps[name])
                          for name in nameslist}
            strategy = Strategy(list(strategies.values()))
            gaps = {name: listintosublist(list(strategy.gapsahead_df[name].values),
                                          strategies[name].lapslist
                                          ) for name in nameslist}
            delta = sum([sum([abs(x - y)
                              for x, y in zip([item for sublist in prevgaps[name] for item in sublist],
                                              list(strategy.gapsahead_df[name].values))
                              ])
                         for name in nameslist])
        return strategies, strategy


engine = create_engine(
    'mysql://mf6bshg8uxot8src:nvd3akv0rndsmc6v@nt71li6axbkq1q6a.cbetxkdyhwsb.us-east-1.rds.amazonaws.com:3306/'
    'ss0isbty55bwe8te')
eventoptions = [{'label': item, 'value': item} for item in sorted(list(
    pandas.read_sql_query("SELECT DISTINCT session_id FROM `Calendar` WHERE session='R1' or session='R2'",
                          engine).session_id))]
tyreoptions = []

USERNAMES = [['Alvaro', 'AlvaroFormoso789'],
             ['Miguel', 'MiguelFabra159'],
             ['Josete', 'IlCapo528'],
             ['David', 'ArreuCambero936'],
             ['Adrian', 'Adrivolador102'],
             ['Jorge', 'Lamoraldelalcoiano520'],
             ['Mario', 'Cambi789'],
             ['Andrea', 'Rochetto741'],
             # ['Jan','JanSumann123'],
             # ['Dani','DaniRincon456'],
             ]

app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(app, USERNAMES)
server = app.server
app.layout = html.Div([html.H2(id='title',
                               children='Campos Engineering',
                               style={'text-align': 'center',
                                      'fontSize': 20}),
                       html.Div([html.Div(dcc.Dropdown(id='eventselector',
                                                       multi=False,
                                                       value='',
                                                       options=eventoptions,
                                                       placeholder="Select Event",
                                                       style={'height': '30px'}),
                                          style={'display': 'inline-block',
                                                 'width': '30%',
                                                 'height': '30px',
                                                 'border-style': 'none'}),
                                 html.Div(dcc.Dropdown(id='tyreselector',
                                                       multi=True,
                                                       value='',
                                                       options=tyreoptions,
                                                       placeholder="Select Tyre",
                                                       style={'height': '30px'}),
                                          style={'display': 'inline-block',
                                                 'width': '30%',
                                                 'height': '30px',
                                                 'border-style': 'none'}),
                                 html.Div(dcc.Input(id='stopsselector',
                                                    value=0,
                                                    type='number',
                                                    placeholder="maximum number of pit-stops",
                                                    style={'font-size': '18px',
                                                           'width': '100%',
                                                           'height': '30px'}),
                                          style={'display': 'inline-block',
                                                 'width': '30%',
                                                 'height': '30px',
                                                 'vertical-align': 'top',
                                                 'border-style': 'none'}),
                                 daq.BooleanSwitch(id='overtakingswitch',
                                                   on=False,
                                                   label="Overtaking Model",
                                                   labelPosition="left",
                                                   color="rgb(0,255,0)",
                                                   style={'fontsize': 20,
                                                          'width': '9%',
                                                          'float': 'right',
                                                          'vertical-align': 'top'})
                                 ]),
                       html.Div('',
                                style={'width': '100%',
                                       'height': '15px'}),
                       dcc.Tabs(id="tabs",
                                children=[dcc.Tab(id='Tyres', value='tab-1', label='Tyres'),
                                          dcc.Tab(id='Raceplot', value='tab-2', label='Raceplot'),
                                          dcc.Tab(id='Results', value='tab-3', label='Results'),
                                          ],
                                value='tab-1')
                       ])


if __name__ == '__main__':
    app.run_server(debug=False)


@app.callback([Output('Tyres', 'children'),
               Output('Raceplot', 'children'),
               Output('Results', 'children')],
              [Input('title', 'n_clicks'),
               Input('tabs', 'value'),
               Input('eventselector', 'value'),
               Input('tyreselector', 'value'),
               Input('stopsselector', 'value'),
               Input('overtakingswitch', 'value')])
def update_all(title, tabs, eventselector, tyreselector, stopsselector, overtakingswitch):
    event = Event()
    drivers = [Driver(0, 'xxx')]
    tyres = []
    for stops in range(0,int(stopsselector)+1):
        tyres.append(list(itertools.combinations(tyreselector, stops + 1)))

    laps=list([list(seq) for i in range(0,int(stopsselector)) for seq in itertools.permutations(list(range(1, event.laps + 1)), i) if sum(seq) == event.laps])
    for driver in drivers:
        for lap in laps:
            for tyre in tyres:
                strategies = strategies + [DriverStrategy(lap, tyre, driver, event,
                                                          name=event.name + ' ' + driver.name + ' ' + str(
                                                              [t.name for t in tyre]) + ' ' + str(lap))]

    return [], [], []
=======
# import math
# import json
# import time
# import datetime
import numpy as np
import pandas
import itertools
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_daq as daq
from dash.dependencies import Input, Output, State
import dash_auth
# from flask_caching import Cache
# import plotly.graph_objs as go
from sqlalchemy import create_engine
# import os


def listintosublist(primarylist, sublist):
    newlist = []
    for lensublist in sublist:
        newsublist = []
        for i in range(0, lensublist):
            newsublist.append(primarylist.pop(0))
        newlist.append(newsublist)
    return newlist


class Tyre:

    def __init__(self, a, b, c, delta_pace=0, name=''):
        self.name = name
        self.delta_pace = delta_pace
        self.a = a
        self.b = b
        self.c = c

    def eq_model(self, lap, a, b, c):
        return a*lap+b*np.exp(c*lap) + self.delta_pace

    def loss_fun(self, lapsinstint):
        return {lap: self.eq_model(lap, self.a, self.b, self.c) for lap in range(1, lapsinstint + 1)}


class Driver:

    def __init__(self, delta_pace=0, name=''):
        self.delta_pace = delta_pace
        self.name = name
        self.max_time_loss = 0.5
        self.starting_time_loss = 0
        self.gap_at_timeloss0 = 1

    def loss_fun(self, lapsinstint):
        return {lap: self.delta_pace for lap in range(1, lapsinstint + 1)}

    def traffic_loss_fun(self, gap):
        if 0 < gap < self.gap_at_timeloss0:
            return gap / self.gap_at_timeloss0 * (self.starting_time_loss - self.max_time_loss) + self.max_time_loss
        else:
            return 0


class Stint:

    def __init__(self, laps, tyre, driver, refpace, initialloss, gaps):
        self.laps = laps
        self.tyre = tyre
        self.driver = driver
        self.refpace = refpace
        self.initialloss = initialloss
        self.gaps = gaps
        self.laptimes = self.laptimes_fun()

    def laptimes_fun(self):
        tyreloss = self.tyre.loss_fun(self.laps)
        driverloss = self.driver.loss_fun(self.laps)
        laptimes = {lap: tyreloss[lap] + driverloss[lap] + self.refpace + self.driver.traffic_loss_fun(self.gaps[lap-1])
                    for lap in range(1, self.laps + 1)}
        laptimes[1] = laptimes[1] + self.initialloss
        return laptimes


class Event:

    def __init__(self, name=''):
        self.name = name
        self.laps = 30
        self.refpace = 100
        self.pitloss = 30
        self.pitloss_vsc = 25
        self.pitloss_sc = 20
        self.gridloss = 0.1
        self.startingposition = 1  # {'': 1}


class DriverStrategy:

    def __init__(self, lapslist, tyrelist, driver, event, name='', gaps=False):
        self.name = name
        self.lapslist = lapslist
        self.tyrelist = tyrelist
        self.driver = driver
        self.refpace = event.refpace
        self.pitloss = event.pitloss
        self.initialloss = event.gridloss*event.startingposition  # [driver.name]
        if gaps:
            self.gaps = gaps
        else:
            self.gaps = [[2]*laps for laps in lapslist]
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
            stintinstance = Stint(stintlaps, self.tyrelist[stintnumber-1], self.driver, self.refpace, initloss,
                                  self.gaps[stintnumber-1])
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
        self.laptimes_df = self.laptimes_df_fun()
        self.gaps_df = self.gaps_df_fun()
        self.positions_df = self.laptimes_df.cumsum().rank(axis=1, method='dense')
        self.gapsahead_df = self.gapsahead_df_fun()

    def raceplot_fun(self):
        raceplot = pandas.DataFrame()
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

    def laptimes_df_fun(self):
        laptimes_df = pandas.DataFrame()
        for driver_strategy in self.driver_strategylist:
            laptimes_df[driver_strategy.name] = pandas.Series(driver_strategy.laptimes,
                                                              index=driver_strategy.laptimes.keys())
        return laptimes_df

    def gaps_df_fun(self):
        cumulative_df = self.laptimes_df.cumsum()
        gaps_df = cumulative_df.apply(lambda row: row-cumulative_df.min(axis=1))
        return gaps_df

    def gapsahead_df_fun(self):
        racelaps = len(self.laptimes_df.index)
        data = []
        for lap in range(1, racelaps + 1):
            data.append(list(self.laptimes_df.cumsum().sort_values(lap, axis=1).columns))
        data = np.array(data)
        driver_byposition = pandas.DataFrame(data=data,
                                             columns=range(1, len(self.laptimes_df.columns) + 1),
                                             index=range(1, racelaps + 1))
        gapsahead_df = pandas.DataFrame()
        for strategy in self.gaps_df.columns:
            temp = []
            for lap in self.gaps_df.index:
                if self.gaps_df[strategy][lap] == 0:
                    temp.append(0)
                else:
                    myposition = self.positions_df[strategy][lap]
                    mycompetitor = driver_byposition[myposition-1][lap]
                    temp.append(self.gaps_df[strategy][lap]-self.gaps_df[mycompetitor][lap])
            gapsahead_df[strategy] = temp
        return gapsahead_df


class Race:

    def __init__(self, event, strategylist, maxiterations, tolerance):
        self.event = event
        self.maxiterations = maxiterations
        self.tolerance = tolerance
        self.strategies, self.strategy = self.calcrace_fun(strategylist)

    def calcrace_fun(self, strategylist):
        delta = 10 ^ 1000
        i = 0
        gaps = {driverstrategy.name: [[2]*laps for laps in driverstrategy.lapslist] for driverstrategy in strategylist}
        strategies = {driverstrategy.name: driverstrategy for driverstrategy in strategylist}
        nameslist = list(strategies.keys())
        strategy = {}
        while delta > self.tolerance or i < self.maxiterations:
            i = i + 1
            prevgaps = gaps
            strategies = {name: DriverStrategy(strategies[name].lapslist,
                                               strategies[name].tyrelist,
                                               strategies[name].driver,
                                               self.event,
                                               name=name,
                                               gaps=gaps[name])
                          for name in nameslist}
            strategy = Strategy(list(strategies.values()))
            gaps = {name: listintosublist(list(strategy.gapsahead_df[name].values),
                                          strategies[name].lapslist
                                          ) for name in nameslist}
            delta = sum([sum([abs(x - y)
                              for x, y in zip([item for sublist in prevgaps[name] for item in sublist],
                                              list(strategy.gapsahead_df[name].values))
                              ])
                         for name in nameslist])
        return strategies, strategy


engine = create_engine(
    'mysql://mf6bshg8uxot8src:nvd3akv0rndsmc6v@nt71li6axbkq1q6a.cbetxkdyhwsb.us-east-1.rds.amazonaws.com:3306/'
    'ss0isbty55bwe8te')
eventoptions = [{'label': item, 'value': item} for item in sorted(list(
    pandas.read_sql_query("SELECT DISTINCT session_id FROM `Calendar` WHERE session='R1' or session='R2'",
                          engine).session_id))]
tyreoptions = []

USERNAMES = [['Alvaro', 'AlvaroFormoso789'],
             ['Miguel', 'MiguelFabra159'],
             ['Josete', 'IlCapo528'],
             ['David', 'ArreuCambero936'],
             ['Adrian', 'Adrivolador102'],
             ['Jorge', 'Lamoraldelalcoiano520'],
             ['Mario', 'Cambi789'],
             ['Andrea', 'Rochetto741'],
             # ['Jan','JanSumann123'],
             # ['Dani','DaniRincon456'],
             ]

app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(app, USERNAMES)
server = app.server
app.layout = html.Div([html.H2(id='title',
                               children='Campos Engineering',
                               style={'text-align': 'center',
                                      'fontSize': 20}),
                       html.Div([html.Div(dcc.Dropdown(id='eventselector',
                                                       multi=False,
                                                       value='',
                                                       options=eventoptions,
                                                       placeholder="Select Event",
                                                       style={'height': '30px'}),
                                          style={'display': 'inline-block',
                                                 'width': '30%',
                                                 'height': '30px',
                                                 'border-style': 'none'}),
                                 html.Div(dcc.Dropdown(id='tyreselector',
                                                       multi=True,
                                                       value='',
                                                       options=tyreoptions,
                                                       placeholder="Select Tyre",
                                                       style={'height': '30px'}),
                                          style={'display': 'inline-block',
                                                 'width': '30%',
                                                 'height': '30px',
                                                 'border-style': 'none'}),
                                 html.Div(dcc.Input(id='stopsselector',
                                                    value=0,
                                                    type='number',
                                                    placeholder="maximum number of pit-stops",
                                                    style={'font-size': '18px',
                                                           'width': '100%',
                                                           'height': '30px'}),
                                          style={'display': 'inline-block',
                                                 'width': '30%',
                                                 'height': '30px',
                                                 'vertical-align': 'top',
                                                 'border-style': 'none'}),
                                 daq.BooleanSwitch(id='overtakingswitch',
                                                   on=False,
                                                   label="Overtaking Model",
                                                   labelPosition="left",
                                                   color="rgb(0,255,0)",
                                                   style={'fontsize': 20,
                                                          'width': '9%',
                                                          'float': 'right',
                                                          'vertical-align': 'top'})
                                 ]),
                       html.Div('',
                                style={'width': '100%',
                                       'height': '15px'}),
                       dcc.Tabs(id="tabs",
                                children=[dcc.Tab(id='Tyres', value='tab-1', label='Tyres'),
                                          dcc.Tab(id='Raceplot', value='tab-2', label='Raceplot'),
                                          dcc.Tab(id='Results', value='tab-3', label='Results'),
                                          ],
                                value='tab-1')
                       ])


if __name__ == '__main__':
    app.run_server(debug=False)


@app.callback([Output('Tyres', 'children'),
               Output('Raceplot', 'children'),
               Output('Results', 'children')],
              [Input('title', 'n_clicks'),
               Input('tabs', 'value'),
               Input('eventselector', 'value'),
               Input('tyreselector', 'value'),
               Input('stopsselector', 'value'),
               Input('overtakingswitch', 'value')])
def update_all(title, tabs, eventselector, tyreselector, stopsselector, overtakingswitch):
    event = Event()
    drivers = [Driver(0, 'xxx')]
    tyres = []
    for stops in range(0,int(stopsselector)+1):
        tyres.append(list(itertools.combinations(tyreselector, stops + 1)))

    laps=list([list(seq) for i in range(0,int(stopsselector)) for seq in itertools.permutations(list(range(1, event.laps + 1)), i) if sum(seq) == event.laps])
    for driver in drivers:
        for lap in laps:
            for tyre in tyres:
                strategies = strategies + [DriverStrategy(lap, tyre, driver, event,
                                                          name=event.name + ' ' + driver.name + ' ' + str(
                                                              [t.name for t in tyre]) + ' ' + str(lap))]

    return [], [], []
>>>>>>> master
