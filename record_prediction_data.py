# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 10:32:08 2015

@author: ennever

Ping nextbus API every 30s for a particular MBTA bus stop and direction
"""

import requests, time
import xml.etree.ElementTree as ET
import MySQLdb as mdb
import sys
from requests.exceptions import ConnectionError
import pandas as pd




class nextbus_query:
    
   def __init__(self, route = '1', stopID = '0074', agency = 'mbta', table = 'Data_Table_1', db = 'nextbus_1_0074'):
        self.route = route #named route
        self.stopID = stopID #stop ID number from nextbus
        self.agency = agency #nextbus agency
        self.table = table #SQL table to write too
        self.db = db #SQL database
        
        
        self.username = 'ennever'
        self.password = 'ennever123'
        
        self.url = 'http://webservices.nextbus.com/service/publicXMLFeed' #nextbus API url
        self.payload = {'command':'predictions', 'a': self.agency, 'r':self.route, 
           'stopId':self.stopID} #combination of parameters to send to API
   
   
   def create_table(self, table = None, curtype = None, dropif = False):
       if table == None:
           table = self.table
           
       con = self.connect_db()
       with con:
           cur = con.cursor(mdb.cursors.DictCursor)
           if dropif:
               cur.execute("DROP TABLE IF EXISTS " + table)
           cur.execute("CREATE TABLE " + table + "(Id INT PRIMARY KEY AUTO_INCREMENT, \
                 Stop_ID INT, Vehicle INT, Query_Time BIGINT, Predicted_Time BIGINT, \
                 Query_Day CHAR(10))")
           self.table = table
        
   def connect_db(self):
       return mdb.connect('localhost', self.username, self.password, self.db)
       
   def query_nb_api(self):
        try:
            r = requests.get(self.url, params=self.payload)
        except ConnectionError as e:
            print e
            r = 'No response'
            return 0
        else:
            try:
                root = ET.fromstring(r.text)
            except ET.ParseError as pe:
                print pe
                return 0
            self.predictions = root.findall('./predictions/direction/prediction')
            if len(self.predictions) == 0:
                return 0
            else:
                return 1
            
   def record_query(self, debug = False):
        con = self.connect_db()
        with con:
            cur = con.cursor(mdb.cursors.DictCursor)
            cs = ', '
            column_headers = ' (Stop_ID, Vehicle, Query_Time, Predicted_Time, Query_Day) '
            for prediction in self.predictions:
                predictiontime = prediction.get('epochTime')
                seconds = prediction.get('seconds')
                vehicle = prediction.get('vehicle')
                querytime = str(int(predictiontime) - int(seconds)*1000)
                day = time.strftime('%w', time.localtime(1e-3*int(querytime)))
                values = "VALUES(" + str(self.stopID) + cs + str(vehicle) + cs + \
                    str(querytime) + cs + str(predictiontime) + cs + str(day) + ')'
                if debug:
                    print values
                cur.execute("INSERT INTO " + self.table + column_headers + values)
        
            
   def read_queries(self) :
        con = self.connect_db()
        rows = pd.read_sql("SELECT * FROM " + \
                self.table + " ORDER BY Vehicle;", con=con)
        #with con:
        #    cur = con.cursor(mdb.cursors.DictCursor)
        #    cur.execute("SELECT Vehicle, Query_Time, Predicted_Time FROM " + \
        #        self.table + " ORDER BY Vehicle")
        #    rows = cur.fetchall()
        return rows
            
   def start_queries(self, interval = 30, duration = 18): #do a query every "interval" seconds and record it in table
       nqueries = (duration * 3600) / interval
       print 'Querying Nextbus Agency: ' + self.agency + ', Route: ' + self.route + ', Stop: ' + self.stopID
       print 'Recording to MySQL DB: ' + self.db + ', Table: ' + self.table
       print 'Recording for ' + str(duration) + ' hours' 
       print 'Began at: ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
       for i in range(nqueries):
           message = 'Current itteration: ' + str(i) +' of ' + str(nqueries)
           sys.stdout.write('\r' + message)
           print '',
           if i != 0:
               time.sleep(interval)

           if self.query_nb_api() == 1:
               self.record_query()
        
               
#nbq = nextbus_query()
#nbq.create_table(dropif = True)
#nbq.do_query()
#nbq.record_query(debug = True)
#nbq.start_queries(duration = 12) 
#rows = nbq.read_queries()  
#for row in rows[-20:]:
#    print row['Vehicle'], time.strftime('%H:%M:%S', time.localtime(1e-3*row['Predicted_Time'])), 
        

