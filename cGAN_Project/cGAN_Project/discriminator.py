import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, nc, n_conditions, ndf):
        super(Discriminator, self).__init__()
        self.n_conditions = n_conditions

        # Input is Image + Condition channels
        self.main = nn.Sequential(
            # input is (nc + n_conditions) x 256 x 256
            nn.Conv2d(nc + n_conditions, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*32) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, conditions):
        # Expand conditions to match image spatial dimensions
        # conditions: [batch, n_cond] -> [batch, n_cond, 1, 1] -> [batch, n_cond, H, W]
        conditions = conditions.view(conditions.size(0), conditions.size(1), 1, 1)
        conditions = conditions.expand(conditions.size(0), conditions.size(1), input.size(2), input.size(3))
        # Concatenate along channel dimension
        x = torch.cat([input, conditions], 1)
        return self.main(x)
