"""
ODENet model

IMPORTANT FOR TIME-DEPENDANT PARAMETERS
https://stackoverflow.com/questions/41860451/odeint-with-multiple-parameters-time-dependent

TODO Multiple parameters time-dependent odeint
https://stackoverflow.com/questions/41860451/odeint-with-multiple-parameters-time-dependent
https://www.google.com/search?q=ODE+given+variables+over+time&oq=ODE+given+variables+over+time&aqs=chrome..69i57j33i160.17833j0j9&sourceid=chrome&ie=UTF-8
https://www.mathworks.com/matlabcentral/answers/97074-how-do-i-solve-an-ode-with-time-dependent-parameters-in-matlab
https://www.google.com/search?sxsrf=ALeKk031G9mKYVs32pPA80LLlnEW5eFSJQ%3A1607090390489&ei=1kDKX5OhHcaegQbCq5r4Aw&q=torchdiffeq+odeint+time-dependent+variable&oq=torchdiffeq+odeint+time-dependent+variable&gs_lcp=CgZwc3ktYWIQAzoECAAQRzoECCMQJ1Do9AJY1PsCYJT-AmgAcAJ4AIABVIgBmgSSAQE3mAEAoAEBqgEHZ3dzLXdpesgBCMABAQ&sclient=psy-ab&ved=0ahUKEwiT5OKevrTtAhVGT8AKHcKVBj8Q4dUDCA0&uact=5
https://arxiv.org/pdf/2005.00865.pdf
https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import torch
import torch.nn as nn
import torch.optim as optim


class ODENet(nn.Module):
    def __init__(self, dof=3, hidden_nodes=100):
        super(ODENet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dof, hidden_nodes),
            nn.Tanh(),
            nn.Linear(hidden_nodes, dof),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        print('ODENet3D: inputs[{}] y={}'.format(t, y.size()))
        y = self.net(y)
        print('ODENet3D: outputs[{}]'.format(y.size()))
        return self.net(y)

class ODENet3D(nn.Module):
    def __init__(self, dof=3, ode_hidden_nodes=200):
        super(ODENet3D, self).__init__()
        self.dof = dof
        self.ode_hidden_nodes = ode_hidden_nodes

        # ODENet
        self.ode_linear1 = nn.Linear(self.dof, self.ode_hidden_nodes)
        nn.init.normal_(self.ode_linear1.weight, mean=0, std=0.1)
        nn.init.constant_(self.ode_linear1.bias, val=0)
        self.ode_tanh = nn.Tanh()
        self.ode_linear2 = nn.Linear(self.ode_hidden_nodes, self.dof)
        nn.init.normal_(self.ode_linear2.weight, mean=0, std=0.1)
        nn.init.constant_(self.ode_linear2.bias, val=0)

    def forward(self, t, x):
        # print('ODENet3D: inputs[{}] y={}'.format(t, len(x)))
        # tuple (point, EP, plan_dir, window)
        xp = x[0]   # 3dof
        xe = x[1]   # entry point (torch.Size([1, 3]))
        xd = x[2]   # plan dir (torch.Size([1, 3]))
        xw = x[3]   # window
        # print('ODENet3D: inputs[{}] xp={} xe={} xd={} xw={}'.format(t, xp.size(), xe.size(), xd.size(), xw.size()))

        # ODENet
        xp = self.ode_linear1(xp)
        xp = self.ode_tanh(xp)
        xp = self.ode_linear2(xp)
        # print('ODENet3D: outputs[{}]'.format(xp.size()))

        return (xp, xe, xd, xw)

class ODENet3DFull(nn.Module):
    ''' 3dof given tuple (position, EP, plan direction, window) '''
    def __init__(self, dof=3, features_hidden_nodes=None, ode_hidden_nodes=None):
        super(ODENet3DFull, self).__init__()

        channels = 1
        self.dof = dof
        self.features_hidden_nodes = features_hidden_nodes
        self.window_hidden_nodes = 0 #16 * 3 ** 3
        self.ode_hidden_nodes = ode_hidden_nodes

        # NN features (EP, plan_dir)
        self.features_linear1 = nn.Linear(6, self.features_hidden_nodes)
        nn.init.normal_(self.features_linear1.weight, mean=0, std=0.1)
        nn.init.constant_(self.features_linear1.bias, val=0)
        self.features_relu = nn.ReLU()

        # CNN window:  1@9x9x9
        # self.window_conv1 = self._make_conv_layer(channels, 16)  # 16@3x3x3

        # ODENet
        self.ode_linear1 = nn.Linear(self.dof, self.ode_hidden_nodes)
        nn.init.normal_(self.ode_linear1.weight, mean=0, std=0.1)
        nn.init.constant_(self.ode_linear1.bias, val=0)
        self.ode_tanh = nn.Tanh()

        # fully connected
        self.fc1 = nn.Linear(self.features_hidden_nodes + self.window_hidden_nodes + self.ode_hidden_nodes, 32) #128
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.constant_(self.fc1.bias, val=0)
        self.fc2 = nn.Linear(32, 8)   # 128, 64
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        nn.init.constant_(self.fc2.bias, val=0)
        self.fc3 = nn.Linear(8, self.dof)  # 64
        nn.init.normal_(self.fc3.weight, mean=0, std=0.1)
        nn.init.constant_(self.fc3.bias, val=0)
        self.leakyrelu = nn.LeakyReLU()
        # TODO nn.BatchNorm1d(128) and nn.BatchNorm1d(64)
        # self.drop = nn.Dropout(p=0.15)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3,3,3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2,2,2))
        )
        return conv_layer

    def forward(self, t, x):
        xt = t

        # tuple (point, EP, plan_dir, window)
        xp = x[0]   # 3dof
        xe = x[1]   # entry point (torch.Size([1, 3]))
        xd = x[2]   # plan dir (torch.Size([1, 3]))
        xw = x[3]   # window
        # print('ODENet3D: inputs[{}] xp={} xe={} xd={} xw={}'.format(t, xp.size(), xe.size(), xd.size(), xw.size()))

        # features
        xf = torch.cat((xe, xd), dim=2)
        xf = self.features_linear1(xf)
        xf = self.features_relu(xf)
        # print('xe={} xd={} xf={}'.format(xe.size(), xd.size(), xf.size()))
        # xf = xd.view(xf.size(0), -1)

        # window
        # xw = self.window_conv1(xw)
        # xw = xw.view(xw.size(0), -1)

        # ODENet
        xp = self.ode_linear1(xp)
        xp = self.ode_tanh(xp)
        # print('xf={} xp={}'.format(xf.size(), xp.size()))

        # Fully-connected layers: multiple input model architecture (features+window+ode+time)
        x = torch.cat((xf, xp), dim=2)
        x = self.fc1(x)
        x = self.leakyrelu(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.leakyrelu(x)
        # x = self.drop(x)
        x = self.fc3(x)
        # print('ODENet3D: output=[{}]'.format(x.size()))

        return (x, xe, xd, xw)


class WindowDownsampling(nn.Module):

    def __init__(self, window=None, dof=9, plan_hidden_nodes=100, local_hidden_nodes=10, channels=1):
        super(WindowDownsampling, self).__init__()

        # window
        self.window = window

        # multiple input model architecture (window + direction)
        if window == 9:
            # input image:  1@9x9x9
            self.conv_layer1 = self._make_conv_layer(channels, 16)  # 16@3x3x3
            # self.fc1 = nn.Linear(16 * 3 ** 3 + 10, 128)  # 2000

        elif window == 11:
            # input image:  1@11x11x11
            self.conv_layer1 = self._make_conv_layer(channels, 16)  # 16@7x7x7
            self.conv_layer2 = self._make_conv_layer(16, 32)  # 32@3x3x3
            # self.fc1 = nn.Linear(32 * 3 ** 3 + 10, 128)  # 864

        # features  (plan)
        self.plan_features1 = nn.Linear(dof, plan_hidden_nodes)
        self.tanh = nn.Tanh()
        nn.init.normal_(self.plan_features1.weight, mean=0, std=0.1)
        nn.init.constant_(self.plan_features1.bias, val=0)

        # features (local direction)
        self.local_features1 = nn.Linear(3, local_hidden_nodes)
        self.relu = nn.ReLU()
        nn.init.normal_(self.local_features1.weight, mean=0, std=0.1)
        nn.init.constant_(self.local_features1.bias, val=0)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3,3,3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2,2,2))
        )
        return conv_layer

    def forward(self, x):
        # tuple (point/EP/plandir, local_dir, window)
        xp = x[0]
        xd = x[1]
        xw = x[2]

        # plan features
        xp = self.plan_features1(xp)
        xp = self.tanh(xp)

        # local features
        xd = self.local_features1(xd)
        xd = self.relu(xd)

        # window
        if self.window == 9:
            xw = self.conv_layer1(xw)
            # print('     conv_layer1(xw)={}'.format(xw.size()))

        elif self.window == 11:
            xw = self.conv_layer1(xw)
            xw = self.conv_layer2(xw)

        return (xp, xd, xw)


class ODEBlock(nn.Module):
    tol = 1e-3
    adjoint = False

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        if self.adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        print('ODEBlock:: xp={} xd={} xw={}'.format(x[0].shape, x[1].shape, x[2].shape))
        self.integration_time = self.integration_time.type_as(x[0])
        out = odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol)
        print('ODEBlock:: out={}'.format(len(out)))
        print('ODEBlock:: op={} od={} ow={}'.format(out[0].shape, out[1].shape, out[2].shape))
        print('ODEBlock:: op[1]={} od[1]={} ow[1]={}'.format(out[0][1].shape, out[1][1].shape, out[2][1].shape))
        # return out[1]
        return (out[0][1], out[1][1], out[2][1])

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ConcatConv3d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv3d, self).__init__()
        self.conv = nn.Conv3d(dim_in+1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self.conv(ttx)


class ODEWindowFunc(nn.Module):
    nfe = 0

    def __init__(self, plan_hidden_nodes=100, local_hidden_nodes=10, channels=1):
        super(ODEWindowFunc, self).__init__()

        # features  (plan)
        self.plan_features2 = nn.Linear(plan_hidden_nodes, plan_hidden_nodes)
        self.tanh = nn.Tanh()
        nn.init.normal_(self.plan_features2.weight, mean=0, std=0.1)
        nn.init.constant_(self.plan_features2.bias, val=0)

        # features (local direction)
        self.local_features2 = nn.Linear(local_hidden_nodes, local_hidden_nodes)
        self.relu = nn.ReLU()
        nn.init.normal_(self.local_features2.weight, mean=0, std=0.1)
        nn.init.constant_(self.local_features2.bias, val=0)

        # window
        # self.norm1 = nn.GroupNorm(min(8, channels), channels)
        # self.reluw = nn.ReLU(inplace=True)
        self.reluw = nn.LeakyReLU()
        self.conv1 = ConcatConv3d(channels, channels, 3, 1, 1)
        # self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = ConcatConv3d(channels, channels, 3, 1, 1)

    def forward(self, t, x):
        self.nfe += 1

        # print('ODEWindowFunc:: x={}'.format(len(x)))
        # print('ODEWindowFunc:: xp={} xd={} xw={}'.format(x[0].shape, x[1].shape, x[2].shape))

        # latent space tuple (point/EP/plandir, local_dir, window)
        xp = x[0]
        xd = x[1]
        xw = x[2]

        # plan_features: point, EP, plan_dir
        xp = self.plan_features2(xp)
        xp = self.tanh(xp)

        # local_features: direction
        xd = self.local_features2(xd)
        xd = self.relu(xd)

        # window
        # xw = self.norm1(xw)
        xw = self.reluw(xw)
        xw = self.conv1(t, xw)
        # xw = self.norm2(xw)
        xw = self.reluw(xw)
        xw = self.conv2(t, xw)

        # print('ODEWindowFunc:: op={} od={} ow={}'.format(xp.shape, xd.shape, xw.shape))

        return (xp, xd, xw)


class WindowFullyConnected(nn.Module):

    def __init__(self, plan_hidden_nodes=100, local_hidden_nodes=10, channels=1):
        super(WindowFullyConnected, self).__init__()

        # window
        # self.norm1 = nn.GroupNorm(min(8, channels), channels)
        # self.reluw = nn.ReLU(inplace=True)
        self.reluw = nn.LeakyReLU()

        self.fc1 = nn.Linear(channels * 3 ** 3 + plan_hidden_nodes + local_hidden_nodes, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

        self.leakyrelu = nn.LeakyReLU()
        self.batch1 = nn.BatchNorm1d(128)
        self.batch2 = nn.BatchNorm1d(64)
        self.drop = nn.Dropout(p=0.15)

        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        nn.init.constant_(self.fc1.bias, val=0)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        nn.init.constant_(self.fc2.bias, val=0)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.1)
        nn.init.constant_(self.fc3.bias, val=0)

    def forward(self, x):
        # print('WindowFullyConnected:: x={}'.format(len(x)))
        # print('WindowFullyConnected:: x0={} x1={} x2={}'.format(x[0].shape, x[1].shape, x[2].shape))

        # latent space tuple (point/EP/plandir, local_dir, window)
        xp = x[0]
        xd = x[1]
        xw = x[2]

        # window
        # xw = self.norm1(xw)
        xw = self.reluw(xw)

        # views
        xp = xp.view(xp.size(0), -1)
        # print('     view(xp)={}'.format(xp.size()))
        xd = xd.view(xd.size(0), -1)
        # print('     view(xd)={}'.format(xd.size()))
        xw = xw.view(xw.size(0), -1)
        # print('     view(xw)={}'.format(xw.size()))

        # multiple input model architecture (plan_features + local_features + window)
        x = torch.cat((xp, xd, xw), dim=1)
        # print('     cat(xp,xd,xw)={}'.format(x.size()))

        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.batch1(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.leakyrelu(x)
        x = self.batch2(x)
        x = self.drop(x)

        x = self.fc3(x)

        return x