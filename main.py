# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
from torch.utils import data
from torch.utils.data import DataLoader
import h5py
import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from math import log10, sqrt
from PIL import Image
from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn



#device = torch.device('cuda')# if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='PyTorch VDSR')
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--lr", type=float, default=0.1)

global opt, model
opt = parser.parse_args()
print(opt)

# Training settings
batch_size = 100
keep_prob = 0.5

use_cuda = opt.cuda
if use_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")


lr = opt.lr
class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index, :, :, :]).float(), torch.from_numpy(self.target[index, :, :, :]).float()

    def __len__(self):
        return self.data.shape[0]


train_set = DatasetFromHdf5('./train.h5')
train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=64, shuffle=True)
#train_loader = torch.utils.data.DataLoader(dataset=train_set,num batch_size=64, shuffle=True)

#x = Image.open("./dataset/test/input.jpg")



#target_data = transforms.ToTensor()(target)
#target_data = target_data.reshape(1, target_data.size()[0], target_data.size()[1], target_data.size()[2])
#x = x.convert('RGB')
#x_data = transform(x)
#x_data = x_data.reshape(1, x_data.size()[0], x_data.size()[1], x_data.size()[2])

#print(x_data)

criterion = nn.MSELoss()


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out

'''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        #self.conv4 = nn.Conv2d(64, 32, kernel_size=7, padding=3)
        #self.conv5 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

        #nn.init.xavier_uniform_(self.conv1.weight)
        #nn.init.xavier_uniform_(self.conv2.weight)
        #nn.init.xavier_uniform_(self.conv3.weight)
        #nn.init.xavier_uniform_(self.conv4.weight)
        #nn.init.xavier_uniform_(self.conv5.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=keep_prob, training=self.training)
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = F.dropout(x, p=keep_prob, training=self.training)
        #x = F.relu(self.conv4(x))
        x = self.conv3(x)

        return x
'''
def psnr2(pred, gt):
    imdff = pred - gt
    rmse = sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * log10(255.0 / rmse)

model = Net()

# model.to(device)
if use_cuda:
    #torch.cuda.set_device(opt.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

cudnn.benchmark = True
if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()
#model.load_state_dict(torch.load("./models/save1"))
optimizer = optim.Adam(model.parameters(), lr=lr)

def resize_image(img, img_name, pixel=3):
    #img.save("./dataset/original/" + img_name + ".jpg")
    #img_ = img.resize((64, int(img.size[1] / pixel), int(img.size[2] / pixel)), Image.BICUBIC)
   # img_ = img_.resize(64, (img.size[1], img.size[2]))
    #img_.save("./dataset/test/" + img_name + ".jpg")
    return img


def change_lr(optimizer, eopch):
    lr = opt.lr*(0.1**(eopch//10))
    return lr


def train(epoch):
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = lr
    #for param_group in optimizer.param_groups:
    #    print(param_group['lr'])

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        '''
        tensor = []
        for num, (image) in enumerate(data):
            image = transforms.ToPILImage(mode='RGB')(image)
            #print(image.shape)
            #print(image.size[0], image.size[1], image.size[2])
            #resized_data = resize_image(image, str(num))
            #resized_data = transforms.ToTensor()(resized_data)
            tensor.append(data.unsqueeze(0))
        tensor = torch.cat(tensor)
        tensor, data = tensor.to(device), data.to(device)
        '''
        #data = data.to(device)
        data = data.cuda()
       # target = target.to(device)
        target = target.cuda()
        optimizer.zero_grad()
        #output = model(tensor)
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.4)
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data))
        # if batch_idx == 42240:
        #    psnr = 10 * log10(1 / loss.data)
        #    print("PSNR: {:.4f} dB".format(psnr))


def test():
    for i in range(2, 5):
        avg_psnr = 0
        avg_com = 0
        for n in range(1, 6):
            target = Image.open("./dataset/test/test" + str(n) + ".png")
            target_data = target.convert('YCbCr')
            compare = target.convert('YCbCr')
            com_tar, e, r = compare.split()
            com_tar = transforms.ToTensor()(com_tar)
            compare = target.resize((int(target.size[0] / i), int(target.size[1] / i)), Image.BICUBIC)
            compare = compare.resize((target.size[0], target.size[1]), Image.BICUBIC)
            comy, a, b = compare.split()
            comy = transforms.ToTensor()(comy)
            compare = transforms.ToTensor()(compare)
            compare = compare.cuda()
            y, cb, cr = target_data.split()
            model.eval()

            y_data = transforms.ToTensor()(y)
            y_data = y_data.reshape(1, y_data.shape[0], y_data.shape[1], y_data.shape[2])

            realtarget = transforms.ToTensor()(target)
            realtarget = realtarget.cuda()

            input = y.resize((int(y.size[0] / i), int(y.size[1] / i)), Image.BICUBIC)
            input = input.resize((y.size[0], y.size[1]), Image.BICUBIC)
            input_data = transforms.ToTensor()(input)
            input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1], input_data.shape[2])

            data = input_data
            data = data.cuda()

            output = model(data)
            output = torch.squeeze(output, 0)
            output = transforms.ToPILImage()(output.cpu().detach())
            output = Image.merge('YCbCr', [output, cb, cr]).convert('RGB')

            if n == 1:
                output.save('output'+str(i)+'.jpg')
            output = transforms.ToTensor()(output)
            output = output.cuda()
            output[output < 0] = 0
            output[output > 1] = 1
            #compare_mse = criterion(compare, realtarget)
            compare_psnr = psnr2(com_tar, comy)
           # compare_psnr = 10 * log10(1 / compare_mse.data)
            avg_com += compare_psnr
            mse = criterion(output, realtarget)
            psnr = 10 * log10(1 / mse.data)
            avg_psnr += psnr
        psnr = avg_psnr / 5
        avg_psnr = avg_com / 5
        print("BICUBIC *", i, 'PSNR: {:.4f} dB'.format(avg_psnr))
        print("*", i, " PSNR: {:.4f} dB".format(psnr))



for epoch in range(1, 3):
    #train(epoch, lr)
    train(epoch)
    if epoch % 1 == 0:
        test()
    #if epoch % 20 == 0:
    #    lr = lr/10
torch.save(model.state_dict(), "./models/save5")


#torch.save(model.state_dict(), "./models/test4")


"""
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.MSELoss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cuda().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 11):
    train(epoch)
    test()


torch.save(model.state_dict(), "./models/test2")
"""
