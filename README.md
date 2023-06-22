# Benchmarking MNIST Dataset:

The MNIST dataset is quite the state-of-the-art. It was used to understand two types of neural networks, namely fully connected and convolutional neural networks.



## Goals:
1. To get familiar with basic pytorch syntax.
2. Understand the concept of training, validation, and testing. 
3. Use the basic features to create fully connected and convolutional neural networks.
4. Learning how to train the networks on the GPUs, tuning and saving them.
5. Testing them on unseen data.



## Try it yourself:
1. Make sure to make a new environment file using the following command.
```
python -m venv .venv
```
2. If you wish to check it out then run the following commands over git bash or command prompt.
```
git clone https://github.com/Shubham1965/mnist-pytorch.git
pip install -r requirements.txt
```

3. Customize the networks, train them, and test it out on the test images. 




## Test Run:
1. Fully connected neural network (FCNN):
   
If you fix the batch size to 32 as given below,
```
dataset = DataLoader(train, batch_size=32)
```
and the have the following network parameters,
```
  self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,16),
            nn.ReLU(),
            nn.Linear(16,10)
        )
```
and set the learning rate to the following,
```
opt = Adam(clf.parameters(), lr = 1e-3)
```
you should get the following loss metrics

2. Convolutional neural netowork (CNN):
If you fix the batch size to 32 as given below,
```
dataset = DataLoader(train, batch_size=32, shuffle=True)
```
and the have the following network parameters,
```
self.model = nn.Sequential(
            nn.Conv2d(1,32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6),10)
        )
```
and set the learning rate to the following,
```
opt = Adam(clf.parameters(), lr = 1e-3)
```
you should get the following loss metrics
