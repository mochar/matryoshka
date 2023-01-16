import sys
import os

import torch
from torch import nn
from torch.autograd import Variable

from models import *
from data import load_mnist

BATCH_SIZE = 128
LEARNING_RATE = 1e-4 # 1e-3

if __name__ == '__main__':
    model_name = sys.argv[1]
    num_epochs = int(sys.argv[2])
    dataloader, sub0, sub1 = load_mnist(BATCH_SIZE)

    # Pick model
    dim = sub0.shape[0]
    if model_name == 'base':
        sub = sub0
        model = Autoencoder(dim).cuda()
    else:
        sub = sub1
        model = (Jian if model_name == 'jian' else Matryoshka)(dim).cuda()

        # Load parameters from base model
        base_model = Autoencoder(dim).cuda()
        base_model.load_state_dict(torch.load('./models/base.pth'))
        model_state = model.state_dict()
        if model_name == 'jian':
            # Load only decoder parameters
            for k, v in base_model.state_dict().items():
                if k.startswith('decoder.'):
                    model_state[k] = v
        else:
            # Load decoder and encoder parameters
            model_state.update(base_model.state_dict())
        model.load_state_dict(model_state)

    # Continue training
    if os.path.exists(f'./models/{model_name}.pth'):
        c = None
        while c not in ('y', 'n'):
            c = input('Continue training from previous time? (y/n): ')
        if c == 'y':
            model.load_state_dict(torch.load(f'./models/{model_name}.pth'))

    # Train
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE)#, weight_decay=1e-5)
    try:
        for epoch in range(num_epochs):
            for data in dataloader:
                img, _ = data
                img = img.view(img.size(0), -1)
                img = img[:, sub]
                img = Variable(img).cuda()

                output = model(img)
                loss = criterion(output, img)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'epoch [{epoch+1}/{num_epochs}], loss:{loss.data:.4f}')
    except KeyboardInterrupt:
        pass

    # Save
    if not os.path.exists('./models'):
        os.mkdir('./models')
    torch.save(model.state_dict(), f'./models/{model_name}.pth')
