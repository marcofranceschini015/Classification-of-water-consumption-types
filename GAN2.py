import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Suponha que você já tenha carregado seu conjunto de dados e o tenha convertido para tensores PyTorch
# Certifique-se de ajustar isso de acordo com o seu conjunto de dados
# Aqui, estou criando dados de exemplo aleatórios
data = pd.read_csv("train_without_outliers.csv")

# Exemplo de dados
# num_samples = 1000

consumer_type_to_select = "construction"
filtered_df = data[data['Consumer_type'] == consumer_type_to_select]

# Coletando os dados do DataFrame
year = filtered_df['Year'].values  # Substitua 'Year' pelo nome real da coluna do ano
month = filtered_df['Month'].values  # Substitua 'Month' pelo nome real da coluna do mês
consumption = filtered_df['Consumption'].values  # Substitua 'Consumption' pelo nome real da coluna do consumo

# Lidando com colunas categóricas
label_encoder_consumer_number = LabelEncoder()
label_encoder_installation_zone = LabelEncoder()

consumer_number = label_encoder_consumer_number.fit_transform(filtered_df['Consumer_number'].values.ravel())
installation_zone = label_encoder_installation_zone.fit_transform(filtered_df['Installation_zone'].values.ravel())

# Convertendo para tensores PyTorch
year = torch.tensor(year, dtype=torch.float32).view(-1, 1)
month = torch.tensor(month, dtype=torch.float32).view(-1, 1)
consumption = torch.tensor(consumption, dtype=torch.float32).view(-1, 1)
consumer_number = torch.tensor(consumer_number, dtype=torch.long).view(-1, 1)  # Long para dados categóricos
installation_zone = torch.tensor(installation_zone, dtype=torch.long).view(-1, 1)  # Long para dados categóricos

# year = torch.randint(2000, 2023, (num_samples, 1))
# month = torch.randint(1, 13, (num_samples, 1))
# consumption = torch.randint(1, 100, (num_samples, 1))

# consumer_number = [f"Consumer_{i}" for i in torch.randint(1, 6, (num_samples,))]
# installation_zone = [f"Zone_{i}" for i in torch.randint(1, 4, (num_samples,))]

# Convertendo para tensores PyTorch
year = year.float()
month = month.float()
consumption = consumption.float()

# LabelEncoder para converter colunas categóricas em números
le_consumer_number = LabelEncoder()
le_installation_zone = LabelEncoder()

consumer_number = torch.tensor(le_consumer_number.fit_transform(consumer_number))
installation_zone = torch.tensor(le_installation_zone.fit_transform(installation_zone))

# Concatenando todos os tensores
features = torch.cat([year, month, consumption, installation_zone.view(-1, 1)], dim=1)

# Criando um DataLoader com o conjunto de dados original
original_dataset = TensorDataset(features, consumer_number)
original_dataloader = DataLoader(original_dataset, batch_size=32, shuffle=True)

# Modelo GAN simples
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Parâmetros
input_size = features.size(1)
output_size = len(le_consumer_number.classes_)

print(input_size)
latent_size = 10
lr = 0.01
epochs = 50

# Inicialização de modelos, otimizadores e critério
generator = Generator(latent_size, input_size)
discriminator = Discriminator(input_size)
generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()
generator_losses = []
discriminator_losses = []

# Treinamento GAN
for epoch in range(epochs):
    for real_data, _ in original_dataloader:
        batch_size = real_data.size(0)

        # Treinamento do discriminador
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        discriminator_optimizer.zero_grad()

        # Treinamento com dados reais
        real_output = discriminator(real_data)
        real_loss = criterion(real_output, real_labels)
        real_loss.backward()

        # Treinamento com dados gerados
        latent_noise = torch.randn(batch_size, latent_size)
        generated_data = generator(latent_noise)
        fake_output = discriminator(generated_data.detach())
        fake_loss = criterion(fake_output, fake_labels)
        fake_loss.backward()

        discriminator_optimizer.step()

        # Treinamento do gerador
        generator_optimizer.zero_grad()
        generated_output = discriminator(generated_data)
        generator_loss = criterion(generated_output, real_labels)
        generator_loss.backward()
        generator_optimizer.step()
        
        generator_losses.append(generator_loss.item())
        discriminator_losses.append(real_loss.item() + fake_loss.item())

    # Mostrar progresso a cada 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Generator Loss: {generator_loss.item()}, Discriminator Loss: {real_loss.item() + fake_loss.item()}')

# Gerar dados sintéticos
num_synthetic_samples = 100
latent_noise = torch.randn(num_synthetic_samples, latent_size)
synthetic_data = generator(latent_noise)

# Converter de volta para classes originais
generated_consumer_numbers = le_consumer_number.inverse_transform(torch.argmax(synthetic_data, dim=1).numpy())

# Concatenar dados sintéticos com dados originais
synthetic_features = synthetic_data
all_features = torch.cat([features, synthetic_features], dim=0)
all_consumer_numbers = torch.tensor(le_consumer_number.transform(generated_consumer_numbers))

# Agora, "all_features" e "all_consumer_numbers" contêm o conjunto de dados balanceado
# Você pode usar esses dados para treinar seu classificador PyTorch
print("----------------------------------------")
all_features_np = all_features.detach().numpy()
all_consumer_numbers_np = all_consumer_numbers.numpy()

# Criar um DataFrame
columns = ['Year', 'Month', 'Consumption', 'Installation_zone', 'Generated_Consumer_number']
# data = pd.DataFrame(data=synthetic_data.detach().numpy, columns=columns)
newData = pd.DataFrame(data=synthetic_data.detach().numpy(), columns=columns)
newData['Generated_Consumer_number'] = generated_consumer_numbers

# Visualizar o DataFrame
print(newData.head(20))

plt.figure(figsize=(10, 5))
plt.plot(generator_losses, label='Generator Loss')
plt.plot(discriminator_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('GAN Training Losses')
plt.show()

year_tensor = torch.tensor(year, dtype=torch.float32)
month_tensor = torch.tensor(month, dtype=torch.float32)
consumption_tensor = torch.tensor(consumption, dtype=torch.float32)
consumer_number_tensor = torch.tensor(consumer_number, dtype=torch.float32)
installation_zone_tensor = torch.tensor(installation_zone, dtype=torch.float32)

# Calcular a média e o desvio padrão
mean_values = torch.tensor([
    torch.mean(year_tensor),
    torch.mean(month_tensor),
    torch.mean(consumption_tensor),
    torch.mean(consumer_number_tensor),
    torch.mean(installation_zone_tensor)
])

std_values = torch.tensor([
    torch.std(year_tensor),
    torch.std(month_tensor),
    torch.std(consumption_tensor),
    torch.std(consumer_number_tensor),
    torch.std(installation_zone_tensor)
])

# Desnormalização
desnormalized_data = newData * std_values + mean_values

# Resultados
print("Dados Normalizados:")
print(synthetic_data)
print("\nDados Desnormalizados:")
print(desnormalized_data)