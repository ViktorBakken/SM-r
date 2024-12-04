import os
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryAUROC, BinaryAccuracy, BinaryF1Score

#Defining variables
input_size = 15
batch_size = 32
epochs = 250
lr = 0.001
test_size = 0.2
momentum = 0.8
dropout_rate = 0.5

class ANN(nn.Module):
    def __init__(self, n_features):
        super(ANN,self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, 1)   
        )
        
    def forward(self,x):
        return self.network(x)
    
def print_metrics(epoch, avg_train_loss, avg_test_loss):
    print(f"\nEpoch {epoch} done, Training Loss: {avg_train_loss:.4f}, Testing Loss: {avg_test_loss:.4f}\n")
    print("Training metrics:")
    for name, metric in metrics_train.items():
        result = metric.compute()
        print(f"{name}: {result:.4f}")
        metric.reset()
    print("\nTesting metrics:")
    for name, metric in metrics_test.items():
        result = metric.compute()
        print(f"{name}: {result:.4f}")
        metric.reset()
        
def update_metrics(metrics, output, target):
    #Convert output and target into binary
    outputs_binary = (output.squeeze() > 0.5).int()  
    targets_binary = target.squeeze().int()     
    #Update each metric with detached binary outputs and targets.
    for metric in metrics.values():
        metric.update(outputs_binary.detach(), targets_binary.detach())
        
def train_model(epochs, model, training_data, testing_data, optimizer, loss_fn, device):
    avg_loss_train = []
    avg_loss_test = []
    scaler = GradScaler()  #Initialize the gradient scaler
    
    for epoch in range(epochs + 1):
        model.train()
        running_loss_train = 0.0
        num_batches_train = 0
        
        for input, target in training_data:
            input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            #Run the forward pass with autocast
            with autocast():
                output = model(input)
                loss = loss_fn(output, target)
            
            #Backward pass with automatic mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
            #Extract loss for epoch
            running_loss_train += loss.item()
            num_batches_train += 1
            
            #Update metric for later display
            update_metrics(metrics_train, output, target)
        
        #Compute average loss over epoch
        average_train_loss = running_loss_train / num_batches_train
        avg_loss_train.append(average_train_loss)
            
        model.eval()
        running_loss_test = 0.0
        num_batches_test = 0
        with th.no_grad():
            for input, target in testing_data:
                input, target = input.to(device, non_blocking=True), target.to(device, non_blocking=True)
                #Run the forward pass with autocast
                with autocast():
                    output = model(input)
                    loss = loss_fn(output, target)
                
                #Extract loss for epoch
                running_loss_test += loss.item()
                num_batches_test += 1
                
                #Update metrics for later display
                update_metrics(metrics_test, output, target)

        #Compute average loss over epoch
        average_test_loss = running_loss_test / num_batches_test
        avg_loss_test.append(average_test_loss)
        
        #Print metrics once every tenth epoch
        if epoch % 5 == 0:
            print_metrics(epoch, average_train_loss, average_test_loss)
    
    return model, avg_loss_train, avg_loss_test
    
model = ANN(input_size)

#Run model on GPU if possible
device = th.device("cuda" if th.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("Moving model to GPU")
    model.to(device)
else:
    print("Model running on CPU")
    
#Specify cross entropy as loss function 
loss_fn = nn.BCEWithLogitsLoss()  
#Specify SGD as optimizer
#optimizer = th.optim.Adam(model.parameters(), lr = lr)
optimizer = th.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

#Init metrics
metrics_test = {
    "accuracy": BinaryAccuracy().to(device),
    "precision": BinaryPrecision().to(device),
    "recall": BinaryRecall().to(device),
    "f1_score": BinaryF1Score().to(device),
    "roc_auc": BinaryAUROC().to(device)
        }
        
metrics_train = {
    "accuracy": BinaryAccuracy().to(device),
    "precision": BinaryPrecision().to(device),
    "recall": BinaryRecall().to(device),
    "f1_score": BinaryF1Score().to(device),
    "roc_auc": BinaryAUROC().to(device)
        }

#Prepare data

df = pd.read_csv(os.path.abspath("SM-r/data/normalized_labeled_training_data.csv"))

#Select all columns except the last
X = df.iloc[:, :-1]
#Select label column
y = df['increase_stock']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

X_train = th.tensor(X_train.values, dtype=th.float32)
y_train = th.tensor(y_train.values, dtype=th.float32).unsqueeze(1)

X_test = th.tensor(X_test.values, dtype=th.float32)
y_test = th.tensor(y_test.values, dtype=th.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

print("Starting training...")
model, loss_train, loss_test = train_model(epochs=epochs, model=model, 
                                        optimizer=optimizer, loss_fn=loss_fn, 
                                        training_data=train_loader, testing_data=test_loader,
                                        device=device)
print("Training complete")