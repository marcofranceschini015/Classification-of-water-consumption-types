import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.datasets import make_blobs

data = pd.read_csv("train_without_outliers.csv")
data_dim = 5

# Definir o gerador
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
            torch.nn.Linear(2*data_dim, 16),
            torch.nn.BatchNorm1d(16),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(16, data_dim)
        )

    def forward(self, x):
        return self.g(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = torch.nn.Sequential(
            torch.nn.Linear(data_dim, 64),
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

gen = Generator()
disc = Discriminator()
disc.to(device)
gen.to(device)

G_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0002)
D_optimizer = torch.optim.Adam(disc.parameters(), lr=0.0002)

loss = nn.BCELoss()

D_losses = []
G_losses = []

test_data = []

exemplo_consumidores = data[data['Consumer_type'] == 'construction']
exemplo_consumidores = exemplo_consumidores.drop(columns=["Consumer_type"])

epochs = 200
batch_size = 100

dataloader = DataLoader(exemplo_consumidores[:1200], batch_size=batch_size)

for epoch in range(epochs):
    for idx, (true_data, _) in enumerate(dataloader):
        # Training the discriminator
        # Real inputs are actual examples with gaussian distribution
        # Fake inputs are from the generator
        # Real inputs should be classified as 1 and fake as 0
        real_inputs = true_data.to(device)
        real_outputs = disc(real_inputs)
        real_label = torch.ones(real_inputs.shape[0], 1).to(device)

        noise = torch.tensor(np.random.normal(0, 1, (real_inputs.shape[0], 4))).float()
        noise = noise.to(device)
        fake_inputs = gen(noise)
        fake_outputs = disc(fake_inputs)
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

        fake_inputs = gen(noise)
        fake_outputs = disc(fake_inputs)
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
        test_data.append(gen(noise).detach().cpu().numpy())

#plot the loss function
plt.plot(range(len(D_losses)), D_losses)
plt.plot(range(len(G_losses)), G_losses)

plt.ylabel('Loss')
plt.ylabel('batches')
plt.show()




# # Copiar o DataFrame original para manipulação
# exemplo_tipo_mais_frequente = exemplo_consumidores.copy()

# # Inicializar um LabelEncoder para transformar dados categóricos em numéricos
# label_encoders = {}
# categorical_columns = ['Consumer_number', 'Installation_zone']

# for column in categorical_columns:
#     label_encoders[column] = LabelEncoder()
#     exemplo_tipo_mais_frequente[column] = label_encoders[column].fit_transform(exemplo_tipo_mais_frequente[column])

# # Usar esses dados para gerar novos dados baseados no tipo de consumidor mais frequente
# num_samples = 100  # Número de dados a serem gerados
# generated_data_based_on_example = []

# # Gerar dados aleatórios baseados no tipo de consumidor mais frequente
# for _ in range(num_samples):
#     noise = torch.randn(1, data_dim)
#     random_example_index = np.random.randint(0, len(exemplo_tipo_mais_frequente))
#     consumer_info = exemplo_tipo_mais_frequente.iloc[random_example_index][['Year', 'Month', 'Consumption', 'Consumer_number', 'Installation_zone']]
    
#     consumer_info_tensor = torch.from_numpy(consumer_info.values.astype(np.float32)).unsqueeze(0)
    
#     input_to_generator = torch.cat((noise, consumer_info_tensor), dim=1)
    
#     generated_sample = gen(input_to_generator).detach().numpy().flatten()
#     generated_data_based_on_example.append(generated_sample)

# # Converter os dados gerados baseados no exemplo para um DataFrame do Pandas
# df_generated_based_on_example = pd.DataFrame(generated_data_based_on_example, columns=['Year', 'Month', 'Consumption', 'Consumer_number', 'Installation_zone'])

# # Salvar os dados gerados baseados no exemplo em um arquivo CSV
# df_generated_based_on_example.to_csv('dados_gerados_consumidores_baseados_no_exemplo.csv', index=False)
