from __future__ import unicode_literals, print_function, division
import numpy as np
import inspect
import os
import torch
import torch.nn as nn
import random
import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt
import mat73
from constants import *
import re
import time
import math
import datetime

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Mymodel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Mymodel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2m = nn.Linear(input_size, input_size, bias=False)
        nn.init.eye_(self.i2m.weight)
        self.m2m = nn.Linear(input_size, input_size, bias=False)
        nn.init.eye_(self.m2m.weight)
        self.m2o = nn.Linear(input_size, input_size, bias=False)
        nn.init.eye_(self.m2o.weight)
        self.n2n = nn.Linear(1, input_size, bias=False)
        nn.init.ones_(self.n2n.weight)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.h2h.weight)
        self.tanhact = nn.Tanh()

        self.i2r = nn.Linear(input_size, input_size, bias=False)
        nn.init.zeros_(self.i2r.weight)
        self.h2r = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.h2r.weight)
        self.r2o = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.r2o.weight)

        self.i2z = nn.Linear(input_size, input_size, bias=False)
        nn.init.zeros_(self.i2z.weight)
        self.h2z = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.h2z.weight)

        self.h2o = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.h2o.weight)

        self.sigact = nn.Sigmoid()

    def forward(self, x, normal, hidden):
        x = x * (self.n2n(normal.view(1, 1, -1)))
        middle = x + self.h2h(hidden)
        output = self.m2o(self.m2m(self.i2m(middle)))

        r = self.tanhact(self.h2r(hidden) + self.i2r(x))
        z = self.tanhact(self.h2z(hidden) + self.i2z(x))

        output = output + r * self.h2o(hidden)
        output = (1 - z) * output + z * hidden

        hidden = self.tanhact(middle)
        return output, hidden

def main():
    timea = time.strftime('%m%d%H%M')
    makedirs(savefolder)
    filen = inspect.getfile(inspect.currentframe())
    filen = re.split(r'[\\/]', filen)[-1].replace('.py' , '')  # 현재 파일 명
    makedirs(savefolder + '/' + filen)
    f = open(savefolder + '/' + filen + '/' + filen + timea + '.txt', 'w')

    f.write('learning rate: %f\n' %learning_rate)
    f.write('momentum: ({0},{1})\n'.format(momentum[0],momentum[1]))
    f.write('epoch: {0}\n'.format(num_epochs))
    f.write('seed: {0}\n'.format(seed))

    f.write('Start DNN_ContAzi_HRTF!!!\n')
    f.close()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu  # Set the GPU 2 to use

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('Using {} device:', format(device))
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

    # set random seed
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)# build model

    # perfect sweep load #
    mat_file_name = "data/PESQ_multiele_180_12ch_gpu.mat"
    mat_file = scipy.io.loadmat(mat_file_name)
    ps = mat_file['PS']
    print("size of perfect sweep: ", ps.shape)
    ps = torch.from_numpy(ps).float()

    # load observation signal
    mat_file_name = "data/obssig_multiele_180_12ch_gpu.mat"
    mat_file = scipy.io.loadmat(mat_file_name)
    y = mat_file['y']
    print("size of y(observed signal): ", y.shape)

    # load true hrir
    mat_file_name = "data/FABIAN_multiele_180_12ch_gpu.mat"
    mat_file = mat73.loadmat(mat_file_name)
    hrir = mat_file['hrir']
    angle = mat_file['angle']
    print("size of hrir(True head-related impulse response): ", hrir.shape)

    # initial parameters
    est_length = 192  # number of samples for estimated HRIR filter
    total_sample, _ = np.shape(y)
    ns, _ = np.shape(ps)  # the number of sound source

    # preparation of input and output
    C = torch.zeros(total_sample, 1, est_length * ns)
    for i in tqdm(range(total_sample), total=len(range(total_sample)), desc='Observation matrix C generate', leave=True):
        C[i, :, :] = torch.reshape(torch.flip(ps[:, i:est_length + i], dims=[1]), (1, -1))
    C = C.to(device)
    y = torch.from_numpy(y).float().to(device)

    hrir_p_total = np.zeros((est_length, ns, len(y)))

    # model initialize
    input_size = est_length * ns
    hidden_size = est_length * ns
    criterion = nn.MSELoss()

    nsec = 10
    section = int(total_sample/nsec)
    hop = section-2*ns*est_length

    total_time = 0
    j = 0

    while j*hop+section < total_sample:
        y_sec = y[j*hop:j*hop+section]
        C_sec = C[j*hop:j*hop+section]

        model = Mymodel(input_size, hidden_size).to(device)
        print("total parameter count: ", count_parameters(model))  # 사용
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=momentum)

        start = time.time()
        # training
        for i in range(num_epochs):
            pred_H = torch.zeros(est_length * ns, len(y_sec), device=device)
            total_loss = 0
            hidden = torch.zeros(1, 1, hidden_size, device=device)
            pred_h = hidden.reshape(-1, 1)
            for p in range(section):
                pred_H[:, p] = torch.squeeze(pred_h.detach())
                pred = torch.squeeze(torch.mm(C_sec[p], pred_h), dim=1)
                loss = criterion(pred, y_sec[p])
                total_loss = total_loss + loss
                input = C_sec[p]*(y_sec[p]-pred)
                normal = 1/(torch.norm(C_sec[p])**2)
                output, hidden = model(torch.unsqueeze(input, dim=1), normal, hidden)
                pred_h = pred_h + output.reshape(-1, 1)
            total_loss = torch.log(total_loss / total_sample)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        end = time.time()
        taketime = (end - start)
        total_time = total_time + taketime
        result = datetime.timedelta(seconds=taketime)
        print(result)
        hrir_p = np.squeeze(np.reshape(pred_H.cpu().numpy(), (est_length, ns, -1), order='F'))
        if j == 0:
            hrir_p_total[:,:,j*hop:j*hop+section] = hrir_p
        else:
            hrir_p_total[:,:,j * hop + 2*ns*est_length:j * hop + section] = hrir_p[:,:,2*ns*est_length:]
        j = j + 1

    y_sec = y[j * hop:]
    C_sec = C[j * hop:]
    last_sec, _ = np.shape(y_sec)

    model = Mymodel(input_size, hidden_size).to(device)
    print("total parameter count: ", count_parameters(model))  # 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=momentum)

    start = time.time()
    # training
    for i in range(num_epochs):
        pred_H = torch.zeros(est_length * ns, len(y_sec), device=device)
        total_loss = 0
        hidden = torch.zeros(1, 1, hidden_size, device=device)
        pred_h = hidden.reshape(-1, 1)
        for p in range(last_sec):
            pred_H[:, p] = torch.squeeze(pred_h.detach())
            pred = torch.squeeze(torch.mm(C_sec[p], pred_h), dim=1)
            loss = criterion(pred, y_sec[p])
            total_loss = total_loss + loss
            input = C_sec[p] * (y_sec[p] - pred)
            normal = 1 / (torch.norm(C_sec[p]) ** 2)
            output, hidden = model(torch.unsqueeze(input, dim=1), normal, hidden)
            pred_h = pred_h + output.reshape(-1, 1)
        total_loss = torch.log(total_loss / total_sample)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    end = time.time()
    taketime = (end - start)
    total_time = total_time + taketime
    result = datetime.timedelta(seconds=taketime)
    print(result)
    result = datetime.timedelta(seconds=total_time)
    print("total take time: ", result)
    hrir_p = np.squeeze(np.reshape(pred_H.cpu().numpy(), (est_length, ns, -1), order='F'))
    hrir_p_total[:,:,j * hop + 2*ns*est_length:] = hrir_p[:,:,2*ns*est_length:]

    matname = 'save/hrir_p_' + str(int(angle)) + '_' + str(ns) + 'ch_DNN3.mat'
    scipy.io.savemat(matname, {'hrir_p': hrir_p_total})

if __name__ == "__main__":
    main()