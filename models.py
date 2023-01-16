from torch import nn

LATENT_SIZE = 6

def build_coders(dim):
    # encoder = nn.Sequential(
    #     nn.Linear(dim, 128),
    #     nn.ReLU(True),
    #     nn.Linear(128, 64),
    #     nn.ReLU(True), 
    #     nn.Linear(64, 12), 
    #     nn.ReLU(True), 
    #     nn.Linear(12, LATENT_SIZE))
    # decoder = nn.Sequential(
    #     nn.Linear(LATENT_SIZE, 12),
    #     nn.ReLU(True),
    #     nn.Linear(12, 64),
    #     nn.ReLU(True),
    #     nn.Linear(64, 128),
    #     nn.ReLU(True), 
    #     nn.Linear(128, dim), 
    #     nn.Tanh())
    encoder = nn.Sequential(
        nn.Linear(dim, 128),
        nn.ReLU(True), 
        nn.Linear(128, LATENT_SIZE), 
    )
    decoder = nn.Sequential(
        nn.Linear(LATENT_SIZE, 128),
        nn.ReLU(True),
        nn.Linear(128, dim), 
        nn.Tanh())
    return encoder, decoder


class Autoencoder(nn.Module):
    def __init__(self, dim):
        super(Autoencoder, self).__init__()
        self.encoder, self.decoder = build_coders(dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Matryoshka(nn.Module):
    def __init__(self, dim):
        super(Matryoshka, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True))
        self.encoder, self.decoder = build_coders(dim)
        self.decoder0 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True))

        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)

    def forward(self, x):
        x = self.encoder0(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.decoder0(x)
        return x


class Jian(nn.Module):
    def __init__(self, dim):
        super(Jian, self).__init__()
        self.encoder, self.decoder = build_coders(dim)
        self.decoder0 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
            nn.ReLU(True))

        self.decoder.requires_grad_(False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.decoder0(x)
        return x
