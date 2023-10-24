# libraries to import
import time
import os, psutil, socket
from datetime import datetime
from inspect import currentframe, getframeinfo, getouterframes

def __getcurframe__():
    return list(getouterframes(currentframe()))[1]

def __frameinfostr__(frameinfo):
    info = " (" + frameinfo.filename.split('/')[-1] + ":L" + str(frameinfo.lineno) + " " + frameinfo.function + ")"
    return info

def __convert_bytes__(n):
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n

class Timer:
    """
    class to report time taken between creation of timer and current event
    """
    def __init__(self, frameinfo):
        self.refframe = frameinfo 
        self.refdate = datetime.now()
        self.reftime = time.time_ns()
        self.nowframe = None
        self.nowdate = None
        self.nowtime = None


    def get_elapsed_time(self):
        self.nowtime = time.time_ns()
        deltat = float(self.nowtime - self.reftime)/1e9
        return deltat

    def get_elapsed_time_info(self, nowframe):
        self.nowframe = nowframe
        self.nowdate = datetime.now()
        deltat = self.get_elapsed_time()
        info = " - "
        info += __frameinfostr__(self.refframe)
        info += " to "
        info += __frameinfostr__(self.nowframe)
        info += " : Elapsed time (s) " + str(deltat)
        return info 

def Log(message : str, nowframe = __getcurframe__()):
    hostname = socket.gethostname()
    curtime = datetime.now().strftime("[%Y/%m/%d %H:%M:%S] ")
    info = "[" + hostname + "] " + curtime 
    info += __frameinfostr__(nowframe)
    info += ": "
    info += message
    print(info)

def LogElapsedTime(timer : Timer, nowframe = __getcurframe__()):
    Log(timer.get_elapsed_time_info(nowframe), nowframe)

def LogMemory(nowframe = __getcurframe__()):
    memusage = psutil.Process(os.getpid()).memory_full_info()
    info = "Current Memory : "
    info += "RSS=" + __convert_bytes__(memusage.rss) + ", "
    info += "USS=" + __convert_bytes__(memusage.uss) + ", "
    Log(info, nowframe)

def NewTimer(frameinfo = __getcurframe__()):
    return Timer(frameinfo)
