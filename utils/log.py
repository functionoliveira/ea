import os
from datetime import datetime

class Log:
    def __init__(self, path):
        self.path = path
    
    def info(self, message):
        pass
    
class File(Log):
    def __init__(self, path):
        self.path = path
        
    def info(self, message):
        date = datetime.today()
        second = str(date.second).zfill(2)
        minute = str(date.minute).zfill(2)
        hour = str(date.hour).zfill(2)
        day = str(date.day).zfill(2)
        month = str(date.month).zfill(2)
        
        date = f'{day}/{month}/{date.year} {hour}:{minute}:{second}'
        
        if not os.path.isdir(f'{self.path}'):
            os.makedirs(f'{self.path}')
        # if not os.path.isfile(f'{self.path}/info.txt'):
        #     open(f'{self.path}/info.txt', 'x')
        file = open(f'{self.path}/info.txt', 'a+')
        # content = file.readlines()
        # content.append(f'\n{date}: {message}')
        file.write(f'\n{date}: {message}')