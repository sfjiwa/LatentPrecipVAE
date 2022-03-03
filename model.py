import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; 
from scipy.special import gamma, factorial
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):#variational encoder
    def __init__(self, latent_dims):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Dropout2d(p=0.01),
            nn.Conv2d(1,1,3,stride=1,padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(1,1,3,stride=3,padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        #split from previous layer into mean (net 3) and variance (net4)
        self.net3 =  nn.Sequential(
            nn.Conv2d(1,1,3,stride=3,padding=0),
            nn.BatchNorm2d(1), #we can batch normalize the sigma to make it>1
            nn.ReLU(1)#we dont want to constain our mean
        )
        self.net4 =  nn.Sequential(
            nn.Conv2d(1,1,3,stride=3,padding=0),
            nn.BatchNorm2d(1), #we can batch normalize the sigma to make it>1
            nn.ReLU(1)
        )
        self.net5 =  nn.Sequential( #for the skewness of the GEV distribution
            nn.Conv2d(1,1,3,stride=3,padding=0),
            nn.BatchNorm2d(1), #we can batch normalize the sigma to make it>1
            nn.ReLU(1)
        )
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc
        self.N.scale = self.N.scale
        self.N1 = torch.distributions.Normal(0,1)
        self.N1.loc = self.N1.loc
        self.N1.scale = self.N1.scale
        self.kl = 0
    def forward(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h1 = self.net1(xy)
        h2 = self.net2(h1+xy)
        #convOut = torch.flatten(h3,start_dim = 1)
        mu = self.net3(h2)
        sigma = torch.exp(self.net4(h2))
        skew = -0.5*torch.ones(14).to(device)#-1*torch.exp(self.net5(h2))
        z = mu +(torch.exp(torch.lgamma(1-skew)-1))*sigma/skew + (torch.exp(torch.lgamma(1-2*skew))-torch.exp(torch.lgamma(1-skew))**2)*sigma**2/skew**2*self.N.sample(mu.shape).to(device)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).sum()
        return z        


class bgcProx(nn.Module):
    #
    def __init__(self,latent_dims):
        super().__init__()
        self.normalize = nn.BatchNorm2d(1)
        #reshape
        self.lin1 = nn.Linear(106,14)
        self.lin2 = nn.Linear(103,14)
        #self.relu = nn.ReLU()
        #self.par1 = torch.nn.Parameter(torch.zeros(14,14)) #parameters are for element wise matrix function of latent pace proxies
        #self.par2 = torch.nn.Parameter(torch.zeros(14,14)) #they are the size then of the latent space
        #self.par3 = torch.nn.Parameter(torch.zeros(14,14))  # Just use linear if you want it to be precipitation fields 
        #rehape back
    def forward(self,z1,z2=None,y=None): #zy1 must be temperature
        zy = z1 if y is None else torch.cat((z1, y), dim=1)
        #zy2 = z2 if y is None else torch.cat((z2,y),dim=1)
        #zyTemp = self.par1*torch.exp(self.par2*zy1) #the physics part
        #zyLight = self.lin(zy2) + self.par3*zy2 * zy2
        #zy = self.normalize(zyTemp) + self.normalize(zyLight)
        zy = self.lin1 (zy) #for precipitation
        #zy = self.relu(zy)
        zy = torch.transpose(zy, 2, 3) 
        zy = self.lin2 (zy)
        #zy = self.relu(zy)
        return zy


class Decoder(nn.Module): # a layer in the decoder could represent the oceananic fluxes (this could also be in the bcgproxy)
    def __init__(self, latent_dims):
        super().__init__()
        self.circ = torch.nn.Parameter(torch.ones(128))
        self.net1 = nn.Sequential(
            #maybe you will need to unflatten the latent space here
            #nn.Dropout2d(p=0.2),
            nn.ConvTranspose2d(1,1,3,stride=3,padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.ConvTranspose2d(1,1,5,stride=1,padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.ConvTranspose2d(1,1,3,stride=1,padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.ConvTranspose2d(1,1,5,stride=1,padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.ConvTranspose2d(1,1,3,stride=1,padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.ConvTranspose2d(1,1,5,stride=1,padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),

        )
        self.net3 = nn.Sequential(
            nn.ConvTranspose2d(1,1,5,stride=1,padding=2),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.ConvTranspose2d(1,1,5,stride=3,padding=0),
            nn.AdaptiveAvgPool2d((42,42)),
            nn.ConvTranspose2d(1,1,3,padding=1),
            nn.ConvTranspose2d(1,1,5,stride=3,padding=0),
            nn.Dropout2d(p=0.01),
            nn.ConvTranspose2d(1,1,3,stride=1,padding=1),
            
            #this linear layer to represent growth relative to nutrient upwelling/for blending? works without too
            )# its important to note that the linear layer is probabily locationally dependant

    def forward(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        zy = self.net1(zy)
        zy1 = self.net2(zy)
        zy = self.circ*self.net3(zy+zy1) #residual
        return zy


class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    def latentForward(self,x):
        return self.encoder(x)
    def deco(self,x):
        return self.decoder(x)


class PhysicsLatent(nn.Module):
    def __init__(self,latent_dims):
        super(PhysicsLatent, self).__init__()            #might have to change this to 2d
        self.bgcProx = bgcProx(latent_dims)
    def latentForward(self,zy1):
        return self.bgcProx(zy1)


def train(autoencoder, physicslatent,data, slp,epochs=128):
    vae_loss = []
    lin_loss = []
    lam = 1 #parameter lambda
    opt1 = torch.optim.Adam(autoencoder.parameters())
    opt2 = torch.optim.Adam(physicslatent.parameters())

    print(epochs)
    for epoch in range(epochs):
        
        # we train in alternate steps, first fixing h 
        for x,y1 in zip(data,slp): #range data to access both this should def. be batched n done using sgd
            
            x = x.to(device) # GPU
            y1 = y1.to(device)
            opt1.zero_grad()
            x_hat = autoencoder.forward(x)
            x_lin = autoencoder.deco(physicslatent.latentForward(y1))
            loss1 = (((x - x_hat)**2 + (x-x_lin)**2).sum() + 2*autoencoder.encoder.kl)
            loss1.backward()
            
            opt1.step()
            opt2.zero_grad()
            z = autoencoder.latentForward(x)
            z_h = physicslatent.latentForward(y1)
            loss2 = ((z-z_h)**2).sum()
            loss2.backward()
            opt2.step()
        print('iteration:')
        print(epoch)
        lin_loss.append(loss2)
        vae_loss.append(loss1)
        print(loss1)
        print(loss2)
    # actual, vae, bgc, z actual, z proxy

    plt.imshow(x[0][0].detach().cpu().numpy(),vmin=0,vmax=8)
    #idk
    #plt.gca().invert_yaxis()
    plt.gca().invert_yaxis()
    plt.title('original')
    plt.colorbar(shrink=0.9)
    plt.show()
    plt.imshow(x_hat[0][0].detach().cpu().numpy(),vmin=0,vmax=8)
    plt.gca().invert_yaxis()
    plt.title('vae resconstruction')
    plt.colorbar(shrink=0.9)
    plt.show()
    plt.imshow(x_lin[0][0].detach().cpu().numpy(),vmin=0,vmax=8)
    plt.gca().invert_yaxis()
    plt.title('lin resconstruction')
    plt.colorbar(shrink=0.9)
    plt.show()
    plt.imshow(z[0][0].detach().cpu().numpy())
    plt.gca().invert_yaxis()
    plt.title('vae latent space')
    plt.colorbar(shrink=0.9)
    plt.show()
    plt.imshow(z_h[0][0].detach().cpu().numpy())
    plt.gca().invert_yaxis()
    plt.title('lin latent space')
    plt.colorbar(shrink=0.9)
    plt.show()
    plt.imshow(y1[0][0].detach().cpu().numpy())
    plt.gca().invert_yaxis()
    plt.title('pressure')
    plt.colorbar(shrink=0.9)
    plt.show()
    plt.plot(np.log(vae_loss), label='total loss')
    plt.plot(np.log(lin_loss), label='lin loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
    #plt.imshow(y2[0][0].detach().numpy())
    #plt.gca().invert_yaxis()
    #plt.show()
        
    return autoencoder
latent_dims = 196
autoencoder = Autoencoder(latent_dims).cuda()
physicslatent = PhysicsLatent(latent_dims).cuda()
precip = torch.utils.data.DataLoader(torch.load('precip.pt'))
slp = torch.utils.data.DataLoader(torch.load('slp.pt'))
#radData = torch.utils.data.DataLoader(torch.load('radData.pt'))
model = train(autoencoder, physicslatent, precip, slp)

torch.save(autoencoder.state_dict(),"vaeO2.pth")
torch.save(physicslatent.state_dict(),"phyO2.pth")


