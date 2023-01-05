from neuroevo.solution import Solution
from utils.log import Log, File

class Settings:
    solution=Solution
    epochs=0
    maxevals=0
    threshold=0.0
    popsize=0
    generation=0
    debug=False
    log=Log
    dataloader=None
    dataset=None
    identity=''
    
# Default settings for test MNIST handwritten digit recognition
mnist_digit_recognition_settings = Settings()
mnist_digit_recognition_settings.epochs = 20
mnist_digit_recognition_settings.popsize=10
mnist_digit_recognition_settings.maxevals = 200
mnist_digit_recognition_settings.threshold = 0.0001
mnist_digit_recognition_settings.generation = 10
mnist_digit_recognition_settings.debug = True
mnist_digit_recognition_settings.log = File
mnist_digit_recognition_settings.dataset = None
