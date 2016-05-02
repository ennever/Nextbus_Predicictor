# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:01:01 2016

@author: ennever
"""

import time, sys
nqueries = 10
for i in range(nqueries):
           message = 'Current itteration: ' + str(i) +' of ' + str(nqueries)
           sys.stdout.write('\r' + message)
           print '',
           time.sleep(1)