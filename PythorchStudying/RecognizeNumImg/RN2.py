class Net(torch.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = torch.nn.Linear(784, 1024)
    self.fc2 = torch.nn.Linear(1024, 512)
    self.fc3 = torch.nn.Linear(512, 256)
    self.fc4 = torch.nn.Linear(256, 128)
    self.fc5 = torch.nn.Linear(128, 10)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
      x = x.view(-1, 784)
      x = self.relu(self.fc1(x))
      x = self.relu(self.fc2(x))
      x = self.relu(self.fc3(x))
      x = self.relu(self.fc4(x))
      z = self.fc5(x)
      return z

net = Net().to(device)
cel = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)

EPOCHS = 10
for epoch in range(EPOCHS):
  l_sum = 0

  for batch_idx, (x,y) in enumerate(train_loader):
    x, y = x.to(device), y.to(device)
    z = net(x)
    loss = cel(z, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    l_sum += loss.item()

  print(f'Epoch : {epoch+1:3d} / {EPOCHS}',
        f'Loss: {l_sum:0.6f}')
