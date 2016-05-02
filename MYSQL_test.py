# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:32:53 2016

@author: ennever
"""

import _mysql
import sys
import MySQLdb as mdb

try:
    con = _mysql.connect('localhost', 'testuser', 'test123', 'test')
        
    con.query("SELECT VERSION()")
    result = con.use_result()
    
    print "MySQL version: %s" % \
        result.fetch_row()[0]
    
except _mysql.Error, e:
  
    print "Error %d: %s" % (e.args[0], e.args[1])
    sys.exit(1)

finally:
    
    if con:
        con.close()



try:
    con = mdb.connect('localhost', 'testuser', 'test123', 'test');

    cur = con.cursor()
    cur.execute("SELECT VERSION()")

    ver = cur.fetchone()
    
    print "Database version : %s " % ver
    
except mdb.Error, e:
  
    print "Error %d: %s" % (e.args[0],e.args[1])
    sys.exit(1)
    
finally:    
        
    if con:    
        con.close()