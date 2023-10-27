import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([transforms.ToTensor()])
image = Image.open('simimg/test1.png')
tensor = transform(image)
image_arr = tensor.permute(1, 2, 0).numpy()
plt.imshow(image_arr)
plt.axis('off')
plt.show()

mask = Image.open('simimg/mask1.png')
tensor = transform(mask)
image_arr = tensor.permute(1, 2, 0).numpy()
plt.imshow(image_arr)
plt.axis('off')
plt.show()

# Define the neural network architecture
class InpaintingNet(nn.Module):
    def __init__(self):
        super(InpaintingNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64,                kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128,              kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256,             kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512,             kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 4000,            kernel_size=3, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(4000, 512, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(512, 256,  kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128,  kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64,   kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(64, 1,     kernel_size=4, stride=2, padding=1)
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.deconv5(x))
        x = torch.relu(self.deconv4(x))
        x = torch.relu(self.deconv3(x))
        x = torch.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv1(x))
        return x

net = InpaintingNet()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Load the image and mask
img_path = 'simimg/test1.png'
mask_path = 'simimg/mask1.png'
img = Image.open(img_path).convert('RGB')
mask = Image.open(mask_path).convert('RGB')

# Define the transforms to apply to the image and mask
transform_img = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((256)),
    transforms.ToTensor()
])

# Apply the transforms to the image and mask
img_tensor = transform_img(img).unsqueeze(0)
mask_tensor = transform_img(mask).unsqueeze(0)

# Define the neural network and optimizer

# Train the neural network on the given image and mask
for i in range(20):
    print(i)
    output_tensor = net(img_tensor * mask_tensor)
    loss_tensor = nn.MSELoss()(output_tensor * (1 - mask_tensor), img_tensor * (1 - mask_tensor))
    optimizer.zero_grad()
    loss_tensor.backward()
    optimizer.step()

# Generate the inpainted image
inpainted_img_tensor = net(img_tensor * mask_tensor) * (1 - mask_tensor) + img_tensor * mask_tensor

inpainted_img_tensor_arr = inpainted_img_tensor[0].permute(1, 2, 0).detach().numpy()
plt.imshow(inpainted_img_tensor_arr)
plt.axis('off')
plt.show()

# Try a different file
img = Image.open('simimg/test4.PNG')
mask = Image.open('simimg/mask4.png')

img_tensor = transform_img(img).unsqueeze(0)
mask_tensor = transform_img(mask).unsqueeze(0)

inpainted_img_tensor = net(img_tensor * mask_tensor) * (1 - mask_tensor) + img_tensor * mask_tensor

inpainted_img_tensor_arr = inpainted_img_tensor[0].permute(1,2,0).detach().numpy()
plt.imshow(inpainted_img_tensor_arr)
plt.axis('off')
plt.show()


# Now do it with all the images -------------------------------

import pandas as pd
import numpy as np
import random
import os
import torchvision

def hout_conv(hin, stride, padding, dilation, kernel_size):
    res = ((hin + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
    # print(res)
    return res

res = 101
for i in np.arange(1, 6, 1):
    intermediate = hout_conv(res, 2, 1, 1, 3)
    res = np.floor(intermediate)
    print(res)


def hout_deconv(hin, stride, padding, dilation, kernel_size, output_padding):
    res = ((hin-1) * stride - 2 * padding + dilation * (kernel_size-1) + output_padding + 1)
    return res

res = 4
for i in np.arange(1, 6, 1):
    intermediate = hout_deconv(res, 2, 1, 1, 3, 0)
    res = np.floor(intermediate)
    print(res)


hout_deconv(26, 2, 1, 1, 3, 0 )
hout_deconv(51, 2, 1, 1, 3, 0 )

class InpaintingNet(nn.Module):
    def __init__(self):
        super(InpaintingNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64,                kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128,              kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256,             kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512,             kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 4000,            kernel_size=3, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(4000, 512, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv4 = nn.ConvTranspose2d(512, 256,  kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv3 = nn.ConvTranspose2d(256, 128,  kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64,   kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv1 = nn.ConvTranspose2d(64, 1,     kernel_size=3, stride=2, padding=1, output_padding=0)
    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.deconv5(x))
        x = torch.relu(self.deconv4(x))
        x = torch.relu(self.deconv3(x))
        x = torch.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv1(x))
        return x

net = InpaintingNet()
optimizer = optim.Adam(net.parameters(), lr=0.01)

random.seed(123)

sequence = [str(x) for x in np.arange(1, 100, 1)]
img_name = ['test' + x + '.png' for x in sequence]
mask_name = ['mask' + x + '.png' for x in sequence]

file_df = pd.DataFrame({
    'file_no': sequence,
    'img_name': img_name,
    'mask_name': mask_name,
    'train': random.choices([0, 1], weights=[0.2, 0.8], k=len(sequence))
})

class CustomImageDataset(Dataset):
    def __init__(self, df, img_dir, transform_img, transform_mask):
        self.img_dir = img_dir
        self.images = df.img_name.tolist()
        self.masks = df.mask_name.tolist()
        self.transform_img = transform_img
        self.transform_mask = transform_mask
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.img_dir, self.masks[idx])
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        image = self.transform_img(image)
        mask = self.transform_mask(mask)
        return image, mask

transform_img = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((101)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5])
])
transform_mask = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((101)),
    transforms.ToTensor()
])

traindata = CustomImageDataset(file_df.query('train == 1'), 'simimg', transform_img, transform_mask)
testdata = CustomImageDataset(file_df.query('train == 0'), 'simimg',  transform_img, transform_mask)

tensor, mask = traindata.__getitem__(0)
image_arr = tensor.permute(1, 2, 0).numpy()
# plt.imshow(image_arr)
# plt.axis('off')
# plt.show()

train_dataloader = DataLoader(traindata, batch_size = 12, shuffle=True)
test_dataloader = DataLoader(testdata, batch_size = 12, shuffle=True)
img, mask = next(iter(train_dataloader))
img.shape
mask.shape
# plt.imshow(torchvision.utils.make_grid(img,nrow=3,normalize=False).permute(1,2,0).numpy())
# plt.axis('off')
# plt.show()

for epoch in range(2):
    for i,data in enumerate(train_dataloader):
        img_tensor, mask_tensor = data
        optimizer.zero_grad()
        output_tensor = net(img_tensor * mask_tensor)
        loss_tensor = nn.MSELoss()(output_tensor * (1 - mask_tensor), img_tensor * (1 - mask_tensor))
        loss_tensor.backward()
        optimizer.step()

img_tensor, mask_tensor = next(iter(test_dataloader))

output_tensor = net(img_tensor * mask_tensor) * (1 - mask_tensor) + (img_tensor) * mask_tensor
plt.imshow(torchvision.utils.make_grid(output_tensor, nrow=3).permute(1,2,0).detach().numpy())
plt.axis('off')
plt.show()
