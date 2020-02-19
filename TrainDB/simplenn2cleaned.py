import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import copy
from dbclass import TrainDB

n_epochs = 3
batch_size_train = 32
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 60000/(10*batch_size_train)

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('drive/My Drive/mnist/MNIST_data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=False)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('drive/My Drive/mnist/MNIST_data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

db = TrainDB(network,batchfreq=1)

def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    grad_vec = None
    prev_state = copy.deepcopy(network.state_dict())
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward(create_graph=True,retain_graph=True)
    grads = []
    #print(target)
    for param in network.parameters():
        grads.append(param.grad.view(-1))
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grads])
    #print('Norm of grad')
    #print(torch.norm(grad_vec))
    #print(torch.norm(grad_vec/32.0))
    optimizer.step()
    db.step(epoch,batch_idx,prev_state,network,grad_vec,loss.item())
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
  return

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


test()
n_epochs = 2
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

table1 = db.tweight
table3 = db.tdiffnorm
table2 = db.tnorm

table3[['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']][1:].plot()
table2[['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']].plot()
table3[1:].describe()

#db.ithdiffnorm('conv1.weight',2,2)
#db.tweight
db.ithhess_eigenval(1,1)

from hessian_eigenthings_orig.hvp_operator import compute_hessian_eigenthings

criterion = torch.nn.NLLLoss()
eigvec,eigenvalues = compute_hessian_eigenthings(network,train_loader, criterion, full_dataset=False,use_gpu=False)
