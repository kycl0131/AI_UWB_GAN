import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

import models
import numpy as np
from matplotlib import pyplot as plt


is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')
paint_size = 256

standardizator = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((paint_size,paint_size),antialias=True),
                    transforms.Normalize(mean=(0.5),   # 3 for RGB channels이나 실제론 gray scale
                                         std=(0.5))
                    ])  # 3 for RGB channels이나 실제론 gray scale

# MNIST dataset
train_data = dsets.MNIST(root='data/', train=True, transform=standardizator, download=True)
# 처음 400개의 데이터만 추출
subset_indices = range(1000)
subset_data = Subset(train_data, subset_indices)
subset_indices = range(100)

test_data  = dsets.MNIST(root='data/', train=False, transform=standardizator, download=True)
subsettest_data =Subset(test_data, subset_indices)
batch_size = 50

# train_data_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True,num_workers=4)
train_data_loader = torch.utils.data.DataLoader(subset_data, batch_size, shuffle=True,num_workers=4)
test_data_loader  = torch.utils.data.DataLoader(subsettest_data, batch_size, shuffle=True,num_workers=4)
# test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True,num_workers=4)



def imshow(img):
    img = img.reshape(-1,1,paint_size,paint_size)
    img = (img+1)/2    
    img = img.squeeze()
    np_img = img.numpy()
    plt.imshow(np_img, cmap='gray')
    plt.show()

def imshow_grid(img):
    img = img.reshape(-1,1,paint_size,paint_size)
    img = utils.make_grid(img.cpu().detach())
    img = (img+1)/2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

example_mini_batch_img, example_mini_batch_label  = next(iter(train_data_loader))
print(example_mini_batch_img.shape)
imshow_grid(example_mini_batch_img[0:16,:,:])


# # 모델 인스턴스 생성
generator = models.Generator().to(device)


def init_params(model):
    for p in model.parameters():
        if(p.dim() > 1):
            nn.init.xavier_normal_(p)
        else:
            nn.init.uniform_(p, 0.1, 0.2)

init_params(generator)

# # # # Batch SIze만큼 노이즈 생성하여 그리드로 출력하기
for a, b in train_data_loader:
    real_a, real_b = a.to(device), b.to(device)
    wantlabel = torch.full((a.shape[0],1),3).to(device=device, dtype=torch.int)
    img_fake = generator(real_a,real_b,wantlabel)
    imshow_grid(img_fake)
    patch = img_fake.shape[1:4]
    break

   
discriminator = models.Discriminator().to(device)
init_params(discriminator)


for a, b in test_data_loader:
    real_a, real_b = a.to(device), b.to(device,dtype=torch.int)
    wantlabel = torch.full((real_a.shape[0],1),2).to(device=device, dtype=torch.int)
    img_fake = discriminator(real_a,real_b,wantlabel)
    
    
    # imshow_grid(img_fake)
    # print(img_fake)
    
    wantlabel = torch.randint(0,10,(real_a.size(0),1)).squeeze().to(device=device, dtype=torch.int)
    fakedata = generator(real_a,real_b,wantlabel)
    
    discriminator(fakedata, real_b , wantlabel)
    d_outsize = discriminator(real_a,real_b,wantlabel).shape[2]
    t_p_real = torch.mean( discriminator(real_a,real_b,wantlabel).reshape(-1,d_outsize*d_outsize) ,dim=1)
    
    t_p_fake = (torch.mean(discriminator(fakedata, real_b , wantlabel).reshape(-1,d_outsize*d_outsize) ,dim=1))

    p_real = t_p_real.sum()/batch_size

    p_fake = t_p_fake.sum()/batch_size
    
    
    break

print(p_real,p_fake)

