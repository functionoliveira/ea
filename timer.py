import time
import math
from multiprocessing import Process, Value

def timer():
    seconds=0    ###for 15 minutes delay 
    close_time=time.time()+1
    while True:
        if time.time() > close_time:
            seconds += 1
            print('Timer: {}s'.format(seconds), end="\r")
            close_time=time.time()+1

background_process = Process(target=timer)

def start_timer():
    background_process.start()
    
def stop_timer():
    print()
    background_process.terminate()

class Chronometer:
    def __init__(self, debug=False):
        self.counter = Value('i', 0)
        self.debug = debug
        self.message = ""
        
    def set_message(self, value):
        self.message = value
        
    def timer(self, counter, debug):
        close_time = time.time() + 1
        while True:
            if time.time() > close_time:
                counter.value += 1
                close_time = time.time() + 1
                if debug:
                    seconds = counter.value % 60
                    minutes = math.floor(counter.value / 60) % 60
                    hours = math.floor(counter.value / 3600) % 60
                    days = math.floor(counter.value / 86400)
                    seconds = str(seconds).zfill(2)
                    minutes = str(minutes).zfill(2)
                    hours = str(hours).zfill(2)
                    days = str(days).zfill(2)
                    
                    print(f'{self.message}: {days}:{hours}:{minutes}:{seconds}', end="\r")
    
    def start(self):
        self.process = Process(target=self.timer, args=(self.counter,self.debug))
        self.process.start()
        
    def stop(self):
        self.process.terminate()
        self.process.join()
        self.process.close()
    
    def reset(self):
        self.counter.value = 0
    
    def display(self):
        return self.counter.value
