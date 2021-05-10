class train:
  def __init__(self, iter, lr = 0.0002) :
    self.net_D_A = Discriminator().to(device)
    self.net_D_B = Discriminator().to(device)
    self.net_G_A = Generator().to(device)
    self.net_G_B = Generator().to(device)
    self.loss_L1 = nn.L1Loss().to(device)
    self.loss_gan = nn.MSELoss().to(device)

    self.optim_G = torch.optim.Adam(itertools.chain(self.net_G_A.parameters(), self.net_G_B.parameters()), lr=lr, betas=(0.5,0.999)) 
    self.optim_D = torch.optim.Adam(itertools.chain(self.net_D_A.parameters(), self.net_D_B.parameters()), lr=lr,betas=(0.5,0.999))
    
  
  def train_start(self):
    for epoch in range(1, iter+1):
      for i, data in enumerate(train_loader):
        
        real_A = data['A' if AtoB else 'B'].to(device)
        real_B = data['B' if AtoB else 'A'].to(device)
        
        fake_B = self.net_G_A(real_A)
        rec_A = self.net_G_B(fake_B)
        fake_A = self.net_G_B(real_B)
        rec_B = self.net_G_A.(fake_A)
        
        ### train G ###
        set_requires_grad([self.net_D_A, self.net_D_B], False)
        self.optim_G.zero_grad()
        
        loss_G_A_gan = self.loss_gan(self.net_D_A(fake_B), torch.ones_like(fake_B))
        loss_G_B_gan = self.loss_gan(self.net_D_B(fake_A), torch.ones_like(fake_A))
        loss_G_A_L1 = self.loss_L1(rec_A, real_A)
        loss_G_B_L1 = self.loss_L1(rec_B, real_B)
        loss_G = loss_G_A_gan + loss_G_A_L1 + loss_G_B_gan + loss_G_B_L1
        
        loss_G.backward()
        self.optim_G.step()
        
        ### train D ###        
        set_requires_grad([self.net_D_A, self.net_D_B], True)
        self.optim_D.zero_grad()
                
        pred_real_B = self.net_D_A(real_B)
        pred_fake_B = self.net_D_A(fake_B.detach())
        
        loss_D_real_B = self.loss_gan(pred_real_B, torch.ones_like(pred_real_B))
        loss_D_fake_B = self.loss_gan(pred_fake_B, torch.zeros_like(pred_fake_B))
        loss_D_A = (loss_D_real_B + loss_D_fake_B) * 0.5
        
        loss_D_A.backward()
        
        pred_real_A = self.net_D_A(real_A)
        pred_fake_A = self.net_D_A(fake_A.detach())
        
        loss_D_real_A = self.loss_gan(pred_real_A, torch.ones_like(pred_real_A))
        loss_D_fake_A = self.loss_gan(pred_fake_A, torch.zeros_like(pred_fake_A))
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        
        loss_D_B.backward()
        
        self.optim_D.step()
