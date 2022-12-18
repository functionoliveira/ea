import time
import datetime
from torch.nn.functional import nll_loss, cross_entropy
from torchmetrics import Accuracy, F1Score
import math
             
class Train:
    def __init__(self, model, optimizer, train_loader, validation_loader, device=None, fn_loss=cross_entropy, epochs=500):
        self.model = model
        self.optimizer = optimizer
        self.fn_loss = fn_loss
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self._EPOCHS = epochs
        self.accuracy = Accuracy() if device == None else Accuracy().to(device)
        self.f1_score = F1Score(num_classes=10) if device == None else F1Score(num_classes=10).to(device)
        self.batch_size = train_loader.batch_size
        self.batch_number = len(train_loader)
        
    def train_one_epoch(self, epoch):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.train_loader):
            i = i + 1
            start = time.time()
            # Every data instance is an input + label pair
            inputs, labels = data
            self.optimizer.data = data
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.fn_loss(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            last_loss = running_loss / self.batch_size # loss per batch
            last_accuracy = self.accuracy(outputs, labels)
            last_score = self.f1_score(outputs, labels)
            m = self.batch_number / 10
            loader = ''.join(['.' if i >= (p*m) else ' ' for p in range(1, 11)])
            end = time.time()
            print('\033[1A', end='\x1b[2K')
            print('TRAINING EPOCH {}/{}, BATCH {}/{} [{}], LAST LOSS: {:.4f}, LAST ACCURACY: {:.4f}, TIME: {:.1f}s'.format(epoch + 1, self._EPOCHS, i, self.batch_number, loader, last_loss, last_accuracy, end-start))
            running_loss = 0.

        return last_loss
    
    def train(self):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # best_vloss = 1_000_000.
        for epoch in range(self._EPOCHS):
            start = time.time()

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch)

            # We don't need gradients on to do reporting
            self.model.train(False)

            running_vloss = 0.0
            avg_accuracy = 0.0
            for i, vdata in enumerate(self.validation_loader):
                vinputs, vlabels = vdata
                voutputs = self.model(vinputs)
                vloss = self.fn_loss(voutputs, vlabels)
                avg_accuracy += self.accuracy(voutputs, vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            avg_accuracy = avg_accuracy / (i + 1)
            end = time.time()
            print('\033[1A', end='\x1b[2K')
            print('VALIDATING EPOCH {}/{}, AVG LOSS: {:.4f}, AVG ACCURACY: {:.4f}, TIME: {:.1f}s'.format(epoch + 1, self._EPOCHS, avg_vloss, avg_accuracy, end-start))
            print()
    
class SgdTrain:
    def __init__(self, model, optimizer, train_loader, validation_loader, device=None, batch_size=64, fn_loss=cross_entropy, epochs=500):
        self.model = model
        self.optimizer = optimizer
        self.fn_loss = fn_loss
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self._EPOCHS = epochs
        self.accuracy = Accuracy() if device == None else Accuracy().to(device)
        self.f1_score = F1Score(num_classes=10) if device == None else F1Score(num_classes=10).to(device)
        self.batch_size = batch_size
        self.batch_number = int(math.floor(len(self.train_loader) / self.batch_size))
        
    def train_one_epoch(self, epoch):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.fn_loss(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % self.batch_size == self.batch_size - 1:
                last_loss = running_loss / self.batch_size # loss per batch
                last_accuracy = self.accuracy(outputs, labels)
                last_score = self.f1_score(outputs, labels)
                current_batch = int((i + 1) / self.batch_size)
                loader = ''.join(['.' if current_batch > i else ' ' for i in range(self.batch_number)])
                print('\033[1A', end='\x1b[2K')
                print('TRAINING EPOCH {}/{}, BATCH {}/{} [{}], LAST LOSS: {:.4f}, LAST ACCURACY: {:.4f}, LAST F1 SCORE: {:.4f}'.format(epoch + 1, self._EPOCHS, current_batch, self.batch_number, loader,last_loss, last_accuracy, last_score))
                running_loss = 0.

        return last_loss
    
    def train(self):
        # Initializing in a separate cell so we can easily add more epochs to the same run
        # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # best_vloss = 1_000_000.
        for epoch in range(self._EPOCHS):
            #print('EPOCH {}/{}:'.format(epoch + 1, self._EPOCHS))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch)

            # We don't need gradients on to do reporting
            self.model.train(False)

            running_vloss = 0.0
            avg_accuracy = 0.0
            avg_f1_score = 0.0
            for i, vdata in enumerate(self.validation_loader):
                vinputs, vlabels = vdata
                voutputs = self.model(vinputs)
                vloss = self.fn_loss(voutputs, vlabels)
                avg_accuracy += self.accuracy(voutputs, vlabels)
                avg_f1_score += self.f1_score(voutputs, vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            avg_accuracy = avg_accuracy / (i + 1)
            avg_f1_score = avg_f1_score / (i + 1)
            print('\033[1A', end='\x1b[2K')
            print('VALIDATING EPOCH {}/{}, AVG LOSS: {:.4f}, AVG ACCURACY: {:.4f}, AVG F1 SCORE: {:.4f}'.format(epoch + 1, self._EPOCHS, avg_vloss, avg_accuracy, avg_f1_score))
            print()
            # print('LOSS train {} valid {}'.format(avg_loss, avg_vloss), flush=False)

            # Log the running loss averaged per batch
            # for both training and validation
            # writer.add_scalars('Training vs. Validation Loss',
            #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
            #                 epoch + 1)
            # writer.flush()

            # Track best performance, and save the model's state
            # if avg_vloss < best_vloss:
            #     best_vloss = avg_vloss
            #     model_path = 'model_{}_{}'.format(timestamp, epoch)
            #     torch.save(self.model.state_dict(), model_path)
            
            
class LeeaTrain:
    def __init__(self, model, optimizer, train_loader, validation_loader, batch_size=64, fn_loss=cross_entropy, pop_size=100, gen=500):
        self.model = model
        self.optimizer = optimizer
        self.fn_loss = fn_loss
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self._GENERATIONS = gen
        self.population_size = pop_size
        self.accuracy = Accuracy()
        self.f1_score = F1Score(num_classes=10)
        self.batch_size = batch_size
        self.batch_number = int(math.floor(len(self.train_loader) / self.batch_size))
        
    def train_one_generation(self, generation_index):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.fn_loss(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % self.batch_size == self.batch_size - 1:
                last_loss = running_loss / self.batch_size # loss per batch
                last_accuracy = self.accuracy(outputs, labels)
                last_score = self.f1_score(outputs, labels)
                current_batch = int((i + 1) / self.batch_size)
                loader = ''.join(['.' if current_batch > i else ' ' for i in range(self.batch_number)])
                print('TRAINING EPOCH {}/{}, BATCH {}/{} [{}], LAST LOSS: {}, LAST ACCURACY: {}, LAST F1 SCORE: {}'.format(epoch + 1, self._EPOCHS, current_batch, self.batch_number, loader,last_loss, last_accuracy, last_score), end="\r", flush=True)
                running_loss = 0.

        return last_loss
    
    def train(self):
        for gen in range(self._GENERATIONS):
            population = self.optimizer.create_generation()
            print('GENERATION {}:'.format(gen + 1))
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(population)
            # We don't need gradients on to do reporting
            self.model.train(False)
            running_vloss = 0.0
            for i, vdata in enumerate(self.validation_loader):
                vinputs, vlabels = vdata
                voutputs = self.model(vinputs)
                vloss = self.fn_loss(voutputs, vlabels)
                running_vloss += vloss
            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))