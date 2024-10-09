import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms.functional import resize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
import torchvision.models as models
import math
import torchmetrics

Epoch = 300
Lr = 0.00001
batchsize = 256
##读入数据

class PNGDataset(torch.utils.data.Dataset):
    def __init__(self, data_directory, transform=None):
        self.data_directory = data_directory
        self.image_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert('RGB')  # 转换为RGB图像
        if self.transform is not None:
            image = self.transform(image)
        return image


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


data = PNGDataset('./pngdata',transform=transform)
train_size = int(0.8*len(data))
test_size = len(data) - train_size
train_data , test_data = torch.utils.data.random_split(data,[train_size,test_size])


train_loader = DataLoader(train_data,batch_size = batchsize, shuffle=True)
test_loader = DataLoader(test_data,batch_size = batchsize, shuffle=True)


class VGGencoder(nn.Module):
    def __init__(self):
        super(VGGencoder,self).__init__()
        #encoder
        #block 1
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.conv2 = nn.Conv2d(64,64,3,1,1)
        self.pol1 = nn.MaxPool2d(2,2,return_indices=True)

        #block 2
        self.conv3 = nn.Conv2d(64,128,3,1,1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.pol2 = nn.MaxPool2d(2, 2,return_indices=True)

        #block 3
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.pol3 = nn.MaxPool2d(2,2,return_indices=True)

        # block 4
        self.conv8 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv10 = nn.Conv2d(512, 512, 3, 1, 1)
        self.pol4 = nn.MaxPool2d(2, 2,return_indices=True)

        # block 5
        self.conv11 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv12 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, 1)
        self.pol5 = nn.MaxPool2d(2, 2,return_indices=True)

        #decoder
        #block 6
        self.unpol1 = nn.MaxUnpool2d(2,2)
        self.dec1 = nn.ConvTranspose2d(512,512,3,1,1)
        self.dec2 = nn.ConvTranspose2d(512, 512, 3, 1, 1)
        self.dec3 = nn.ConvTranspose2d(512, 512, 3, 1, 1)

        #block 7
        self.unpol2 = nn.MaxUnpool2d(2, 2)
        self.dec4 = nn.ConvTranspose2d(512, 512, 3, 1, 1)
        self.dec5 = nn.ConvTranspose2d(512, 512, 3, 1, 1)
        self.dec6 = nn.ConvTranspose2d(512, 256, 3, 1, 1)

        # block 8
        self.unpol3 = nn.MaxUnpool2d(2, 2)
        self.dec7 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.dec8 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.dec9 = nn.ConvTranspose2d(256, 128, 3, 1, 1)

        # block 9
        self.unpol4 = nn.MaxUnpool2d(2, 2)
        self.dec10 = nn.ConvTranspose2d(128, 128, 3, 1, 1)
        self.dec11 = nn.ConvTranspose2d(128, 64, 3, 1, 1)

        # block 10
        self.unpol5 = nn.MaxUnpool2d(2, 2)
        self.dec12 = nn.ConvTranspose2d(64, 64,3, 1, 1)
        self.dec13 = nn.ConvTranspose2d(64, 3, 3, 1, 1)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x, indice1 = self.pol1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x, indice2 = self.pol2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x, indice3 = self.pol3(x)

        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x, indice4 = self.pol4(x)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x, indice5 = self.pol5(x)

        x = self.unpol1(input=x, indices=indice5)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))

        x = self.unpol2(input=x, indices=indice4)
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        x = F.relu(self.dec6(x))

        x = self.unpol3(input=x, indices=indice3)
        x = F.relu(self.dec7(x))
        x = F.relu(self.dec8(x))
        x = F.relu(self.dec9(x))

        x = self.unpol4(input=x, indices=indice2)
        x = F.relu(self.dec10(x))
        x = F.relu(self.dec11(x))

        x = self.unpol5(input=x, indices=indice1)
        x = F.relu(self.dec12(x))
        x = F.relu(self.dec13(x))

        return x

#加载权重
vgg16 = models.vgg16(pretrained=True)
with torch.no_grad():
    model = VGGencoder()
    model.conv1.weight.data = vgg16.features[0].weight.data
    model.conv1.bias.data = vgg16.features[0].bias.data
    model.conv2.weight.data = vgg16.features[2].weight.data
    model.conv2.bias.data = vgg16.features[2].bias.data
    model.conv3.weight.data = vgg16.features[5].weight.data
    model.conv3.bias.data = vgg16.features[5].bias.data
    model.conv4.weight.data = vgg16.features[7].weight.data
    model.conv4.bias.data = vgg16.features[7].bias.data
    model.conv5.weight.data = vgg16.features[10].weight.data
    model.conv5.bias.data = vgg16.features[10].bias.data
    model.conv6.weight.data = vgg16.features[12].weight.data
    model.conv6.bias.data = vgg16.features[12].bias.data
    model.conv7.weight.data = vgg16.features[14].weight.data
    model.conv7.bias.data = vgg16.features[14].bias.data
    model.conv8.weight.data = vgg16.features[17].weight.data
    model.conv8.bias.data = vgg16.features[17].bias.data
    model.conv9.weight.data = vgg16.features[19].weight.data
    model.conv9.bias.data = vgg16.features[19].bias.data
    model.conv10.weight.data = vgg16.features[21].weight.data
    model.conv10.bias.data = vgg16.features[21].bias.data
    model.conv11.weight.data = vgg16.features[24].weight.data
    model.conv11.bias.data = vgg16.features[24].bias.data
    model.conv12.weight.data = vgg16.features[26].weight.data
    model.conv12.bias.data = vgg16.features[26].bias.data
    model.conv13.weight.data = vgg16.features[28].weight.data
    model.conv13.bias.data = vgg16.features[28].bias.data

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:1'
    else:
        device='cpu'
    return device

device = get_device()


criterion = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(),lr=Lr)


def make_dir():
    image_dir = 'outimage'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

def save_decod_img(img,epoch):
    img = img.view(img.size(0),3,128,128)
    save_image(img,'./outimage/Autoencoder_image.png'.format(epoch))

def training(model, train_loader, test_loader, Epochs):
    train_loss = []
    test_loss = []
    train_psnr = []
    test_psnr = []
    for epoch in range(Epochs):
        running_loss = 0.0
        for data in train_loader:
            img = data
            img = img.to(device)
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(epoch + 1, Epochs, loss))

        # Calculate train PSNR
        train_mse = torch.mean((img - outputs)**2)
        if train_mse == 0:
            train_psnr_value = float('inf')
        else:
            train_psnr_value = 10 * math.log10(1 / train_mse.item())
        train_psnr.append(train_psnr_value)
        print('Epoch {} of {}, Train PSNR: {:.3f}'.format(epoch + 1, Epochs, train_psnr_value))

        test_running_loss = 0.0
        with torch.no_grad():
            for test_data in test_loader:
                test_img = test_data
                test_img = test_img.to(device)
                test_outputs = model(test_img)
                test_loss_batch = criterion(test_outputs, test_img)
                test_running_loss += test_loss_batch.item()
        test_loss_epoch = test_running_loss / len(test_loader)
        test_loss.append(test_loss_epoch)
        print('Epoch {} of {}, Test Loss: {:.3f}'.format(epoch + 1, Epochs, test_loss_epoch))

        # Calculate test PSNR
        test_mse = torch.mean((test_img - test_outputs)**2)
        if test_mse == 0:
            test_psnr_value = float('inf')
        else:
            test_psnr_value = 10 * math.log10(1 / test_mse.item())
        test_psnr.append(test_psnr_value)
        print('Epoch {} of {}, Test PSNR: {:.3f}'.format(epoch + 1, Epochs, test_psnr_value))

        if epoch % 1 == 0:
            save_decod_img(outputs.cpu().data, epoch)
    return train_loss, train_psnr, test_loss, test_psnr


model.to(device)

make_dir()
train_loss = training(model,train_loader,test_loader,Epochs=Epoch)
torch.save(model,'reconstruction model 1.pth')