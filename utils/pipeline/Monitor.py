

class Overfit:
    def __init__(self, start_epoch = 10, threshold = 0.003, patience = 3):
        self.start_epoch = start_epoch
        self.threshold = threshold
        self.patience = patience
        self.violation_count = 0

    def check(self, epoch, train_loss, val_loss):
        '''
            returns True if the model has overfitted
            returns False if nothing has gone wrong
        '''

        if(epoch >= self.start_epoch and val_loss - train_loss > self.threshold):
            self.violation_count += 1
            print(f"Possible Overfitting!!! {self.violation_count}/{self.patience}\n")

        if(self.violation_count >= self.patience):
            print("Training Stopped!!!")
            return True
        
        return False