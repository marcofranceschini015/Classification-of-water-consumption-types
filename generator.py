import matplotlib.pyplot as plt
#importing Libraries
import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import make_blobs
from scipy.stats import multivariate_normal

X1, Y1 = make_blobs(n_samples=500, centers=[(5, 5)], n_features=3, random_state=0)

fig = plt.figure(figsize=(18, 8))

#Parameters to set
mu_x = 5
variance_x = 0.5

mu_y = 5
variance_y = 0.5

#Create grid and multivariate normal
x = np.linspace(3, 7, 20)
y = np.linspace(3, 7, 20)
X, Y = np.meshgrid(x, y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X;
pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, rv.pdf(pos) * 3, rstride=1, cstride=1, linewidth=1, antialiased=False, cmap='viridis')
ax2 = fig.add_subplot(1, 2, 2)

ax2.scatter(X1[:, 0], X1[:, 1], marker="o", c=Y1, s=25, edgecolor="k")
cc = plt.Circle((5, 5), 1, fill=False, edgecolor='red', linewidth=2)
ax2.set_aspect(1)
ax2.add_artist(cc)

plt.show()

def generate_norm_data(batch_size: int = 16):
    X1, Y1 = make_blobs(n_samples=batch_size, centers=[(5, 5)], n_features=3)
    return X1


from sklearn.datasets import make_blobs

X1 = generate_norm_data(1000)

plt.figure(figsize=(8, 8))
ax2 = plt.gca()

plt.title("One informative feature, one cluster per class", fontsize="small")
ax2.scatter(X1[:, 0], X1[:, 1], marker="o", s=25, edgecolor="k")
cc = plt.Circle((5, 5), 1, fill=False, edgecolor='red', linewidth=2)
ax2.set_aspect(1)
ax2.add_artist(cc)

plt.show()

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class NormDataset(Dataset):
    def __init__(self, n_samples=1000):
        self.Xs, self.y = make_blobs(n_samples=n_samples, centers=[(5, 5)], n_features=3)

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        image = self.Xs[idx].astype(np.float32)
        label = self.y[idx]
        return image, label
    
class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.g = torch.nn.Sequential(
            torch.nn.Linear(4, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.g(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = torch.nn.Sequential(
            torch.nn.Linear(2, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        res = self.d(x)
        return res

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

epochs = 1000
batch_size = 1000

training_data = NormDataset(n_samples=10000)
dataloader = DataLoader(training_data, batch_size=batch_size)

G = Generator()
D = Discriminator()
D.to(device)
G.to(device)

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)

loss = nn.BCELoss()

D_losses = []
G_losses = []

test_data = []

for epoch in range(epochs):
    for idx, (true_data, _) in enumerate(dataloader):
        # Training the discriminator
        # Real inputs are actual examples with gaussian distribution
        # Fake inputs are from the generator
        # Real inputs should be classified as 1 and fake as 0
        real_inputs = true_data.to(device)
        real_outputs = D(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1).to(device)

        noise = torch.tensor(np.random.normal(0, 1, (real_inputs.shape[0], 4))).float()
        noise = noise.to(device)
        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)

        outputs = torch.cat((real_outputs, fake_outputs), 0)
        targets = torch.cat((real_label, fake_label), 0)

        D_loss = loss(outputs, targets)
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Training the generator
        # For generator, goal is to make the discriminator believe everything is 1
        noise = torch.tensor(np.random.normal(0, 1, (real_inputs.shape[0], 4))).float()
        noise = noise.to(device)

        fake_inputs = G(noise)
        fake_outputs = D(fake_inputs)
        fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
        G_loss = loss(fake_outputs, fake_targets)
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        G_losses.append(G_loss.item())
        D_losses.append(D_loss.item())

    if (epoch + 1) % 100 == 0:
        print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}'.format(epoch, idx, D_loss.item(), G_loss.item()))
        test = (torch.rand(real_inputs.shape[0], 4) - 0.5) / 0.5
        noise = torch.tensor(np.random.normal(0, 1, (real_inputs.shape[0], 4))).float().to(device)
        test_data.append(G(noise).detach().cpu().numpy())

#plot the loss function
plt.plot(range(len(D_losses)), D_losses)
plt.plot(range(len(G_losses)), G_losses)

plt.ylabel('Loss')
plt.ylabel('batches')

#noise = torch.randn(size=(500, 4)).cuda()
#noise = (torch.rand(real_inputs.shape[0], 4) - 0.5) / 0.5
#noise = torch.tensor(np.random.normal(0, 1, (500, 4))).float().cuda()
#print(noise.shape)

fig = plt.figure(figsize=(18, 30))

#generated_data = G(noise).detach().cpu().numpy()
for i, generated_data in enumerate(test_data):
    plt.subplot(8, 4, i + 1)
    ax2 = plt.gca()

    plt.title("Epoc %d" % i, fontsize="small")
    ax2.scatter(generated_data[:, 0], generated_data[:, 1], marker="o", s=25, edgecolor="k")
    cc = plt.Circle((5, 5), 1, fill=False, edgecolor='red', linewidth=2)
    ax2.set_aspect(1)
    ax2.add_artist(cc)

plt.show()