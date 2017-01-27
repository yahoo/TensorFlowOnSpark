# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from multiprocessing.managers import BaseManager
from multiprocessing import JoinableQueue

class TFManager(BaseManager): pass

mgr = None
qdict = {}
kdict = {}

def _get(key):
    return kdict[key]

def _set(key, value):
    kdict[key] = value

def start(authkey, queues, mode='local'):
    """
    Create a new multiprocess.Manager (or return existing one).
    """
    global mgr, qdict, kdict
    qdict.clear()
    kdict.clear()
    for q in queues:
      qdict[q] = JoinableQueue()
    TFManager.register('get_queue', callable=lambda qname: qdict[qname])
    TFManager.register('get', callable=lambda key: _get(key))
    TFManager.register('set', callable=lambda key, value: _set(key, value))
    if mode == 'remote':
        mgr = TFManager(address=('',0), authkey=authkey)
    else:
        mgr = TFManager(authkey=authkey)
    mgr.start()
    return mgr

def connect(address, authkey):
    """
    Connect to a multiprocess.Manager
    """
    TFManager.register('get_queue')
    TFManager.register('get')
    TFManager.register('set')
    m = TFManager(address, authkey=authkey)
    m.connect()
    return m

