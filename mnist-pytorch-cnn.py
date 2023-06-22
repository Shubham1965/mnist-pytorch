import torch 
from PIL import Image
import matplotlib.pyplot as plt

from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root="data", download = True, train = True, transform = ToTensor())
dataset = DataLoader(train, batch_size=32, shuffle=True)
# 1,28,28

class ImageClassifier_cnn(nn.Module):
    def __init__(self):
        super().__init__()
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

    def forward(self, x):
        return self.model(x)


clf = ImageClassifier_cnn().to('cuda')
opt = Adam(clf.parameters(), lr = 1e-3)
loss_fn = nn.CrossEntropyLoss()

def test():
    # Load and test the model on images:
    with open('model_state_cnn.pt', 'rb') as f:
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


# Training flow
if __name__ == "__main__":
    is_test = 0 # change this to 1 if you want to test your saved network

    if not is_test:
        losses = []  # Store loss values

        # Training loop:
        for epoch in range(10):
            epoch_loss = 0.0

            for batch in dataset:
                X,y = batch 
                X,y = X.to('cuda'), y.to('cuda')
                yhat = clf(X)
                loss = loss_fn(yhat,y)

                # Apply backprop
                opt.zero_grad()
                loss.backward()
                opt.step()

            epoch_loss /= len(dataset)  # Average loss per batch
            losses.append(epoch_loss)
            print(f"Epoch: {epoch+1}, loss: {loss}")

        # Plot loss metrics
        plt.plot(range(1, len(losses) + 1), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig('CNN_loss_metrics.png')  # Save the plot as a PNG file
        plt.show()

        with open('model_state_cnn.pt', 'wb') as f:
            save(clf.state_dict(), f)
    else:
        test()
    
