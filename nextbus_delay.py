# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:29:46 2016

@author: ennever

methods to calculate the delay of each "trip" versus the query time, and 
put it into a new table. It does so with the following steps:

1. Read in MySQL table of nextbus queries of predicted times and query times
2. Separate out data by Vehicle ID and then separate by original predicted arrival 
time, these will be separate "trips"
 
"""

import pandas as pd
import MySQLdb as mdb
import matplotlib as plot
from record_prediction_data import nextbus_query
import sys
"""
helper function to determine whether the time was "pre_rush", "morning_rush", 
"midday", "evening_rush", "post_rush" or "weekend". demarcations is the time 
that divides the time categories (doesn't apply to weekend). 
"""
def timeofday(time, demarcations = [7, 9.5, 16.5, 18.5]):
    demarcations.sort()
    
    if time.dayofweek in [5,6]:
        return 'weekend'
    else:
        timehour = time.hour + time.minute/60.0
        if timehour <= demarcations[0]:
            return 'pre_rush'
        if (timehour >= demarcations[0]) and (timehour <= demarcations[1]):
            return 'morning_rush'
        if (timehour >= demarcations[1]) and (timehour <= demarcations[2]):
            return 'midday'
        if (timehour >= demarcations[2]) and (timehour <= demarcations[3]):
            return 'evening_rush'
        if timehour >= demarcations[3]:
            return 'post_rush'
        
    
class nextbus_delay:
    
    def __init__(self, query_table = 'Data_Table_1', db = 'nextbus_1_0074', delay_table = 'Delay_Table_1'):
        self.query_table = query_table #SQL table to read from
        self.db = db #SQL database
        self.delay_table = delay_table #SQL table to write to
        
        self.username = 'ennever'
        self.password = 'ennever123'
        self.rows = pd.DataFrame()
        self.delay_df = pd.DataFrame([])
        self.final_delays_df = pd.DataFrame([])
        
    def connect_db(self):
       return mdb.connect('localhost', self.username, self.password, self.db)
        
    def create_delay_table(self, delay_table = None, curtype = None, dropif = False):
       if delay_table == None:
           delay_table = self.delay_table
           
       con = self.connect_db()
       with con:
            cur = con.cursor(mdb.cursors.DictCursor)
            if dropif:
               cur.execute("DROP TABLE IF EXISTS " + delay_table)
            cur.execute("CREATE TABLE " + delay_table + "(Id INT PRIMARY KEY AUTO_INCREMENT, \
                 Stop_ID INT, Vehicle INT, Query_Time BIGINT, Initial_Prediction BIGINT, \
                 Query_Day CHAR(10), Curent_Time_Delta BIGINT)")
            self.delay_table = delay_table
        
    
    def read_query_table(self):
        con = self.connect_db()
        rows = pd.read_sql("SELECT * FROM " + \
                self.query_table + " ORDER BY Vehicle;", con=con)
        rows['Predicted_Time'] = pd.to_datetime(rows['Predicted_Time'], unit = 'ms')
        rows['Query_Time'] = pd.to_datetime(rows['Query_Time'], unit = 'ms')
        self.rows = rows
        return True
    
    def calculate_delays(self):#calculate the delay
        if len(self.rows) == 0:
            self.read_query_table()
        vids = self.rows['Vehicle'].unique() #vehicle IDs
        
        #debug line
        #vids = vids[0]
        delay_df = pd.DataFrame([])
        nvids = len(vids)
        i = 1
        for vid in vids: #separate out data by vehicle ID, then by a particular trip
            message = 'Current itteration: ' + str(i) +' of ' + str(nvids)
            sys.stdout.write('\r' + message)
            print '',
            i += 1
            vehicle_data = self.rows[self.rows.Vehicle == vid]
            delta = pd.DataFrame({'Query_Time':vehicle_data.Query_Time, 'Time_Delta':vehicle_data.Predicted_Time})
            delta = delta.sort(columns = 'Query_Time')
            delta['Time_Delta'] = delta['Time_Delta'].diff() #calculate the difference between sequential predictions
            #delta = delta[delta.Time_Delta != pd.Timedelta(0)] #no reason to keep the zeros
            delta['Time_Delta'] = delta['Time_Delta'].astype('timedelta64[s]')
            #get the indexes of where the time delta is greater than 30 minutes
            tripbegins = delta.index[abs(delta['Time_Delta']) > 1800]
            #now have the indexes of where a particular bus arrives
            #now it's possible to get the index of where a trip begins and when it ends
            #therefore we can get a measure of the cumulative delay
            tripbegin = delta.index[0]
            initial_prediction = vehicle_data['Predicted_Time'].loc[tripbegin]
          
            for delta_index, delta_row in delta.iterrows():
                if (delta_index == tripbegins).any(): #if you're at an initial 
                    cumulative_delay = pd.Timedelta(0)
                    tripbegin = delta_index
                    initial_prediction = vehicle_data['Predicted_Time'].loc[tripbegin]
                    query_time = delta['Query_Time'].loc[tripbegin]
                else:
                    cumulative_delay = vehicle_data['Predicted_Time'].loc[delta_index] - initial_prediction
                    query_time = delta['Query_Time'].loc[delta_index]
                toarrival = initial_prediction - query_time
                newrow = {'Vehicle_ID':vid, 'Trip_Index':tripbegin, 'Initial_Prediction':initial_prediction, 'Query_Time':query_time, 'Cumulative_Delay':cumulative_delay, 'Time_To_Initial_Prediction':toarrival}
                delay_df = delay_df.append(newrow, ignore_index=True)
                
        delay_df['Time_To_Initial_Prediction'] = delay_df['Time_To_Initial_Prediction'].astype('timedelta64[s]')
        delay_df['Cumulative_Delay'] = delay_df['Cumulative_Delay'].astype('timedelta64[s]')
        
        self.delay_df = delay_df
    
    def calc_final_delays(self):#calculte the final delay for a particular trip
        if self.delay_df.size == 0:
            self.calculate_delays()
        
        tripends = self.delay_df.groupby(['Trip_Index'])['Query_Time'].idxmax()
        final_delays_df = self.delay_df.loc[tripends]
        final_delays_df = final_delays_df.loc[:,['Query_Time', 'Initial_Prediction', 'Cumulative_Delay','Vehicle_ID']]
        final_delays_df.rename(columns = {'Cumulative_Delay':'Final_Delay'}, inplace = True)
        final_delays_df = final_delays_df[final_delays_df['Final_Delay'] != 0]
        final_delays_df.loc[:,'Time_Of_Day'] = pd.Series(final_delays_df['Initial_Prediction'].apply(timeofday), index = final_delays_df.index, dtype = 'category')
        self.final_delays_df = final_delays_df
        
    def plot_delays(self):
        self.final_delays_df['Final_Delay'].hist(by=self.final_delays_df['Time_Of_Day'], figsize = (9, 6), sharex = True);
        
        
nbd = nextbus_delay()
nbd.read_query_table()
nbd.calculate_delays()
nbd.calc_final_delays()
nbd.plot_delays()
        
            