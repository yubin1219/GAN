
def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
class train:
  def __init__(self, iter, lr = 0.0002) :
    self.net_D = Discriminator().to(device)
    self.net_G = UnetGenerator().to(device)
    self.loss_L1 = nn.L1Loss().to(device)
    self.loss_gan = nn.BCELoss().to(device)

    self.optim_G = torch.optim.Adam(self.net_G.parameters(), lr=lr, betas=(0.5,0.999)) 
    self.optim_D = torch.optim.Adam(self.net_D.parameters(), lr=lr,betas=(0.5,0.999))
    
  
  def train_start(self):
    for epoch in range(1, iter+1):
      for i, data in enumerate(train_loader):
        
        GT = data['label'].to(device)
        I = data['input'].to(device)
        
        set_requires_grad(self.net_D, True)
        self.optim_D.zero_grad()
        
        output = self.net_G(I)
        real = torch.cat([I, GT], 1)
        fake = torch.cat([I, output], 1)
        
        pred_real = self.net_D(real)
        pred_fake = self.net_D(fake.detach())
        
        loss_D_real = self.loss_gan(pred_real, torch.ones_like(pred_real))
        loss_D_fake = self.loss_gan(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        
        loss_D.backward()
        self.optim_D.step()
        
        set_requires_grad(self.net_D, False)
        self.optim_G.zero_grad()
        
        pred_fake = self.net_D(fake)
        loss_G_gan = self.loss_gan(pred_fake, torch.ones_like(pred_fake))
        loss_G_L1 = self.loss_L1(output, GT)
        loss_G = loss_G_gan + loss_G_L1
        loss_G.backward()
        self.optim_G.step()
        
