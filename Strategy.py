# import math
# import json
import time
import webbrowser
# import datetime
import numpy as np
import pandas
import itertools
# from flask_caching import Cache
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
# import os
import random


def plot_df(df, title='', xaxis='', yaxis=''):
    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(go.Scatter(x=[x+1 for x in list(df.index)],
                                 y=df[column],
                                 mode='lines',
                                 name=column))
    fig.update_layout(title=title,
                      xaxis_title=xaxis,
                      yaxis_title=yaxis)
    return fig


def boxplot_df(df, xcolumn, ycolumn, title='', xaxis='', yaxis=''):
    fig = go.Figure()
    for series in df[xcolumn].unique():
        fig.add_trace(go.Box(y=list(df[df[xcolumn]==series][ycolumn]),
                             name=series))
    fig.update_layout(title=title,
                      xaxis_title=xaxis,
                      yaxis_title=yaxis)
    return fig


def plot_scenario(race, name):
    fig = make_subplots(rows=3, cols=2,
                        shared_xaxes=False,
                        vertical_spacing=0.06,
                        specs=[[{"type": "scatter"},
                                {"type": "table"}],
                               [{"type": "scatter"},
                                {"type": "scatter"}],
                               [{"type": "scatter", "colspan": 2},
                                None]],
                        subplot_titles=('RacePlot', 'Summary', 'Power of Undercut', 'Time Difference'))
    for column in race.strategy.raceplot.columns:
        fig.add_trace(go.Scatter(x=[x+1 for x in list(race.strategy.raceplot.index)],
                                 y=race.strategy.raceplot[column],
                                 mode='lines',
                                 name=column),
                      row=1, col=1)
    summary = race.summary().sort_values(by='position')
    summary['Strategy'] = summary.index
    summary[['avgtime', 'racetime', 'gap_winner']] = summary[['avgtime', 'racetime', 'gap_winner']].round(3)
    fig.add_trace(go.Table(header={'values': list(summary.columns),
                                   'font': {'size': 10},
                                   'align': 'left'},
                           cells={'values': [summary[k].tolist() for k in summary.columns],
                                  'align': 'left'}),
                  row=1, col=2)
    for column in race.strategy.undercut.columns:
        fig.add_trace(go.Scatter(x=[x+1 for x in list(race.strategy.undercut.index)],
                                 y=race.strategy.undercut[column],
                                 mode='lines',
                                 name=column),
                      row=2, col=1)
    for column in race.strategy.compounddeltatime.columns:
        fig.add_trace(go.Scatter(x=[x + 1 for x in list(race.strategy.compounddeltatime.index)],
                                 y=race.strategy.compounddeltatime[column],
                                 mode='lines',
                                 name=column),
                      row=2, col=2)
    for tyre in race.strategy.tyres:
        fig.add_trace(go.Scatter(x=list(range(1,31)),
                                 y=list(tyre.loss_fun(30).values()),
                                 mode='lines',
                                 name=tyre.name),
                      row=3, col=1)
    fig.update_layout({'height':1350,
                       'yaxis2': {'title': '<< Undercut - Overcut >>',
                                  'range': [-5,5]},
                       'yaxis3': {'title': 'delta time [s]',
                                  'range': [-10,10]},
                       'yaxis4': {'title': 'time loss [s]',
                                  'range': [0,10]}})
    return fig


def listintosublist(primarylist, sublist):
    newlist = []
    for lensublist in sublist:
        newsublist = []
        for i in range(0, lensublist):
            newsublist.append(primarylist.pop(0))
        newlist.append(newsublist)
    return newlist


def flatten_list(mylist):
    flatten = lambda l: [item for sublist in l for item in sublist]
    return flatten(mylist)


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def get_table(table, column=None, value=None):
    if (column is None) and (value is None):
        df = pandas.read_sql_query("SELECT * FROM `" + table + "`", engine)
    else:
        df = pandas.read_sql_query("SELECT * FROM `" + table + "` WHERE `" + column + "` LIKE '" + value + "'", engine)
    if len(df) > 0:
        return df
    else:
        print("No data.")


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
        return {i: self.eq_model(i, self.a, self.b, self.c) for i in range(1, lapsinstint + 1)}


class Driver:

    def __init__(self, delta_pace=0, name='', startingposition=0):
        self.delta_pace = delta_pace
        self.name = name
        self.startingposition = startingposition
        self.max_time_loss = 0.5
        self.starting_time_loss = 0
        self.gap_at_timeloss0 = 1

    def loss_fun(self, lapsinstint):
        return {i: self.delta_pace for i in range(1, lapsinstint + 1)}

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
        self.laps = int(self.calendar().race_laps.max())
        self.refpace = self.tyremodels().TyrePace.min()
        self.pitloss = float(self.calendar().pit_loss.mean())
        self.pitloss_vsc = float(self.calendar().pit_loss_vsc.mean())
        self.pitloss_sc = float(self.calendar().pit_loss_vsc.mean())
        self.gridloss = float(self.calendar().grid_loss.mean())
        self.fuel_penalty = float(self.calendar().fuel_penalty.mean())
        self.tyres = list(self.tyremodels().tyre)

    def calendar(self):
        return get_table('Calendar', 'session_id', self.name)


    def tyremodels(self):
        df = get_table('TyreModels', 'Session', self.name)
        bpace = df.TyrePace.min()
        df['tyre'] = df.apply(lambda row: Tyre(row.A, row.B, row.C, row.TyrePace - bpace, row.Compound), axis=1)
        return df


class DriverStrategy:

    def __init__(self, lapslist, tyrelist, driver, event, name='', gaps=False):
        self.name = name
        self.lapslist = lapslist
        self.tyrelist = tyrelist
        self.driver = driver
        self.refpace = event.refpace
        self.pitloss = event.pitloss
        self.initialloss = event.gridloss*driver.startingposition
        self.fuel_penalty = event.fuel_penalty
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
        for lap in range(0, len(laptimes_list)):
            laptimes_list[lap] = laptimes_list[lap]-self.fuel_penalty*lap
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
        self.racetimes = dict(zip([strat.name for strat in driver_strategylist],
                                  [strat.racetime for strat in driver_strategylist]))
        self.avgtimes = dict(zip([strat.name for strat in driver_strategylist],
                                 [strat.avgtime for strat in driver_strategylist]))
        self.winner = min(self.racetimes, key=self.racetimes.get)
        self.quickestavg = min(self.avgtimes.values())
        self.undercut = self.undercut_fun(20)
        self.compounddeltatime = self.compounddeltatime_fun(20)
        self.raceplot = self.raceplot_fun()
        self.laptimes_df = self.laptimes_df_fun()
        self.gaps_df = self.gaps_df_fun()
        self.positions_df = self.laptimes_df.cumsum().rank(axis=1, method='dense')
        self.gapsahead_df = self.gapsahead_df_fun()
        self.tyres = list(set(flatten_list([x.tyrelist for x in driver_strategylist])))

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
        while delta > self.tolerance and i < self.maxiterations:
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

    def summary(self):
        df = pandas.DataFrame.from_dict(self.strategy.avgtimes, orient='index', columns=['avgtime'])
        df2 = pandas.DataFrame.from_dict(self.strategy.racetimes, orient='index', columns=['racetime'])
        df = pandas.concat([df, df2], axis=1)
        df['gap_winner'] = self.strategy.gaps_df.iloc[-1]
        df['position'] = df.avgtime.rank()
        df['starting_position'] = [x.driver.startingposition for x in self.strategies.values()]
        df['position_gain'] = df.position - df.starting_position
        df.reset_index(level=0, inplace=True)
        df.rename(columns={'index': 'name'}, inplace=True)
        return df


engine = create_engine(
    'mysql://mf6bshg8uxot8src:nvd3akv0rndsmc6v@nt71li6axbkq1q6a.cbetxkdyhwsb.us-east-1.rds.amazonaws.com:3306/'
    'ss0isbty55bwe8te')


# if __name__ == '__main__':
print('Loading available events...: ')
[print(x) for x in intersection(list(get_table('Calendar').session_id.unique()),
                                list(get_table('TyreModels').Session.unique()))]
eventname = str(input('Input event name from the available events: '))
stopn = int(input('Input max number of stops: '))
repeat_tyres = input('Can we repeat tyre compound: ').lower() in ['true', '1', 'y', 'yes', 'si', 's']
first_stop_window = int(input('In which lap does the pit-window open? '))
number_of_cars = int(input('How many cars are racing? '))
print('Computing....')


starttime = time.time()
event = Event(eventname)
drivers = [Driver(0, 'xxx')]
tyres = event.tyres

calcoptionstime = time.time()
tyreoptions = []
if repeat_tyres:
    for stops in range(1, int(stopn) + 1):
        tyreoptions = tyreoptions + [list(item) for item in (list(itertools.product(tyres, repeat = stops + 1)))]
else:
    for stops in range(1, int(stopn) + 1):
        tyreoptions = tyreoptions + [list(item) for item in (list(itertools.permutations(tyres, stops + 1)))]
laps = list([list(seq) for i in range(0,int(stopn+2))
             for seq in itertools.permutations(list(range(1, event.laps + 1)),i) if sum(seq) == event.laps])

print('options elapsed time: ' + str(time.time()-calcoptionstime))

strategiestime = time.time()
strategies = []
for driver in drivers:
    for lap in laps:
        if lap[0] >= first_stop_window:
            for tyre in tyreoptions:
                if len(lap)==len(tyre):
                    strategies = strategies + [DriverStrategy(lap, tyre, driver, event,
                                                              name=event.name + ' ' + driver.name + ' ' + str(
                                                                  [t.name for t in tyre]) + ' ' + str(lap))]

print(str(len(strategies)) + ' strategies elapsed time: ' + str(time.time()-strategiestime))

racetime = time.time()
myrace = Race(event, strategies, 1, len(strategies)*0.1)

print('race elapsed time: ' + str(time.time()-racetime))
print('Best strategy: ' + str(list(
    myrace.strategy.positions_df[myrace.strategy.positions_df <= 1].loc[event.laps].dropna().index)))

racetime = time.time()
winnerstrategies_names = list(
    myrace.strategy.positions_df[myrace.strategy.positions_df <= 5].loc[event.laps].dropna().index)
winnerstrategies = [strat for strat in strategies if strat.name in winnerstrategies_names]

drivers = [Driver(0,'pos '+ str(startingpos), startingpos) for startingpos in range(1, number_of_cars + 1)]

myrace = []
for iter in range(0,10000):
    strategies = []
    for driver in drivers:
        singlestrategy = random.choice(winnerstrategies)
        strategies.append(DriverStrategy(singlestrategy.lapslist, singlestrategy.tyrelist, driver, event,
                                                              name=event.name + ' ' + driver.name + ' ' + str(
                                                                  [t.name for t in singlestrategy.tyrelist]) + ' ' +
                                                                   str(singlestrategy.lapslist)))
    myrace.append(Race(event, strategies, 5, len(strategies) * 0.1))
summary_list = [x.summary() for x in myrace]
summary_df = summary_list[0].append(summary_list[1:])

# myrace_winners = Race(event, winnerstrategies, 1, len(winnerstrategies)*0.1)
#
# print('race winners time: ' + str(time.time()-racetime))
#
# racetime = time.time()
# myrace_winners_traffic = Race(event, winnerstrategies, 100, len(winnerstrategies)*0.1)
#
# print('race traffic elapsed time: ' + str(time.time()-racetime))
# print('Best strategy with traffic: ' + str(list(
#     myrace.strategy.positions_df[myrace_winners_traffic.strategy.positions_df <= 1].loc[event.laps].dropna().index)))

# plot_scenario(myrace,'All').write_html('All_cases.html')
# plot_scenario(myrace_winners,'Best').write_html('Best_cases.html')
# plot_scenario(myrace_winners_traffic,'Best with overtaking model').write_html('Best_cases_Overtaking.html')
# # webbrowser.open('All_cases.html')
# webbrowser.open('Best_cases.html')
# webbrowser.open('Best_cases_Overtaking.html')
print('total elapsed time: ' + str(time.time()-starttime))
