import torch
import torch.nn as nn
import torch.optim as optim

from CAE import Conv3dAE


def train(model, epochs, batch_size, learning_rate, dataset):
    outputs = []
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in range(0, )

    return outputs

if __name__ == "__main__":
    model = Conv3dAE()
    dataset = 
    train(model, epochs=10, batch_size=4, learning_rate=0.001, dataset=dataset)