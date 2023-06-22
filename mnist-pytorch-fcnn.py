import torch 
from PIL import Image

from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root="data", download = False, train = True, transform = ToTensor())
dataset = DataLoader(train, 32)
# 1,28,28

class ImageClassifier_fcnn(nn.Module):
    def __init__(self):
        super().__init__()
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

    def forward(self, x):
        return self.model(x)

clf = ImageClassifier_fcnn().to('cuda')
opt = Adam(clf.parameters(), lr = 1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    
    # Training loop:
    for epoch in range(20):
        for batch in dataset:

            X,y = batch 
            # X = X.reshape(X.shape[0],28*28)
            X, y = X.to('cuda'), y.to('cuda')
            yhat = clf(X)
            loss = loss_fn(yhat,y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch: {epoch+1}, loss: {loss}")


    with open('model_state_fcnn.pt', 'wb') as f:
        save(clf.state_dict(), f)

    # Load and test the model on images:
    with open('model_state_fcnn.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    img = Image.open('img_1.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

    print(torch.argmax(clf(img_tensor)))

    img = Image.open('img_2.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

    print(torch.argmax(clf(img_tensor)))

    img = Image.open('img_3.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

    print(torch.argmax(clf(img_tensor)))