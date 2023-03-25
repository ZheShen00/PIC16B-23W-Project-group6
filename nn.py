import torch
from torch.utils.data import TensorDataset, DataLoader

# this file defined a neural network from scratch

def create_data(X_train, y_train, X_val, y_val, X_test, y_test):
    '''
    create tensors for the training, validation, and testing datasets
    '''
    # X_train, X_val, X_test = X_train.to_numpy(), X_val.to_numpy(), X_test.to_numpy()
    # y_train, y_val, y_test = y_train.to_numpy(), y_val.to_numpy(), y_test.to_numpy()
    X_train, X_val, X_test = torch.Tensor(X_train), torch.Tensor(X_val), torch.Tensor(X_test)
    y_train = torch.Tensor(np.array([ [y] for y in y_train ]))
    y_val = torch.Tensor(np.array([ [y] for y in y_val ]))
    y_test = torch.Tensor(np.array([ [y] for y in y_test ]))

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, train_batch_size=16, test_batch_size=32):
    '''
    create dataloaders for train, validation and test sets
    '''

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size = train_batch_size, shuffle = False)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val, y_val), batch_size = test_batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size = test_batch_size, shuffle = False)
    
    return train_loader, val_loader, test_loader

def evaluate_loss(model, criterion, dataloader):
    '''
    calculate the loss given the model
    '''
    model.eval()
    total_loss = 0.0
    for batch_X, batch_y in dataloader:
        batch_size = len(batch_X)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        total_loss += loss.item()

    return total_loss / len(dataloader.dataset)

def evaluate_acc(model, dataloader):
    '''
    calculate the accuracy scores given the model
    '''
    model.eval()
    best_mse = np.inf 
    loss_fn = torch.nn.MSELoss() 
    for batch_X, batch_y in dataloader:
        y_pred = model(batch_X)
        mse = loss_fn(y_pred, batch_y)
        # mse = float(mse)
        if mse.item() < best_mse:
            best_mse = mse.item()
        
    return best_mse
######################################################################
# OneLayerNetwork -- ALSO A LOGISTIC REGRESSION MODEL
######################################################################

class OneLayerNetwork(torch.nn.Module):
    def __init__(self, input_features):
        '''
        implement OneLayerNetwork with torch.nn.Linear. Use sigmoid as the activation
        '''
        super(OneLayerNetwork, self).__init__()
        self.oneLayer = torch.nn.Linear(input_features, 1)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        '''
        implement the foward function
        '''
        l1 = self.oneLayer(x)
        outputs = self.activation(l1)
        return outputs
    
def init_oneLayerNN(in_features, lr):
    '''
    input_features: int, number of input features
    lr: float, learning rate
    prepare the OneLayerNetwork model, criterion, and optimizer
    '''
    
    model = OneLayerNetwork(in_features)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    return model, criterion, optimizer

######################################################################
# TwoLayerNetwork
######################################################################

class TwoLayerNetwork(torch.nn.Module):
    def __init__(self, input_features, hidden_features, first_activation='sigmoid'):
        # 
        '''
        input_features: int, number of input features
        hidden_features: int, size of the hidden layer
        first_activation: str, activation to use for the first hidden layer
        implement TwoLayerNetwork with torch.nn.Linear. Use sigmoid as the activation for both layers
        '''
        super(TwoLayerNetwork, self).__init__()
        
        self.layer1 = torch.nn.Linear(input_features, hidden_features)
        self.layer2 = torch.nn.Linear(hidden_features, 1)
        if first_activation == 'sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif first_activation == 'ReLU':
            self.activation = torch.nn.ReLU()

    def forward(self, x):
        '''
        implement the foward function
        '''
        l1 = self.layer1(x)
        l1_activated = self.activation(l1)
        l2 = self.layer2(l1_activated)
        outputs = self.activation(l2)

        return outputs
    
def init_twoLayerNN(in_features, hidden_size, first_activation, lr):
    '''
    input_features: int -> Number of input features
    hidden_features: int -> Size of the hidden layer
    first_activation: str -> Activation to use for the first hidden layer
    lr: float -> Learning Rate
    prepare the TwoLayerNetwork model, criterion, and optimizer
    '''
    
    model = TwoLayerNetwork(in_features, hidden_size, first_activation)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)

    return model, criterion, optimizer


def train(model, criterion, optimizer, train_loader, valid_loader, num_epochs, logging_epochs=1):
    '''
    Build the training paradigm - Zero out gradients, forward pass, compute loss, loss backward, update model
    and calculate the loss in training and validation and the training and validation accuracy
    '''
    print("Start training model...")

    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    
    for epoch in range(1, num_epochs+1):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad() # zero out gradients
            y_pred = model.forward(batch_X) # forward pass
            loss = criterion(y_pred, batch_y) # compute loss
            loss.backward() # loss backward
            optimizer.step() # update model
            
        train_loss = evaluate_loss(model, criterion, train_loader)
        valid_loss = evaluate_loss(model, criterion, valid_loader)
        train_acc = evaluate_acc(model, train_loader)
        valid_acc = evaluate_acc(model, valid_loader)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)

        if logging_epochs > 0 and epoch % logging_epochs == 0:
            print(f"| epoch {epoch:2d} | train loss mean {train_loss:.6f} | train lost best {train_acc:.6f} | valid loss mean {valid_loss:.6f} | valid loss best {valid_acc:.6f} |")

    return train_loss_list, valid_loss_list, train_acc_list, valid_acc_list