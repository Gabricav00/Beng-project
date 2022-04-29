"""
Regression using ODENet

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim

from Framework.Tools import FileSystem
from Framework.Regression.RegressionModel import RegressionModel
from Framework.Regression.ODENet import ODENet
from Framework.Regression.ODENet import ODENet3D
from Framework.Regression.ODENet import ODENet3DFull
from Framework.Regression.RunningAverageMeter import RunningAverageMeter


class ODERegression(RegressionModel):
    # attributes
    delay = 0.01
    #dof = 3
    method = 'dopri5'
    # batch_time = 8
    batch_size = 1  #100
    num_workers = 6
    niters = 100  #1000
    val_freq = 5    #25
    viz = True
    gpu = 0
    adjoint = False
    t_scale = 0.01
    ode_hidden_nodes = 100
    transfer_knowledge = False
    data_filter = False
    data_augment = True
    planned_dir = True
    learning_rate = 0.001
    weight_decay = 1e-3
    step_decay = 20

    checkpoint_dir = None
    best_val_loss = None
    best_val_itr = None

    def __init__(self, dof=None):
        super().__init__()
        self.dof = dof

        # sys path
        sys.path.append('C:\\UCL\\PhysicsSimulation\\Python\\torchdiffeq')

        # gpu support
        self.device = torch.device('cuda:' + str(self.gpu) if torch.cuda.is_available() else 'cpu')

        # checkpoint_dir
        self.filesystem = FileSystem.FileSystem()
        self.checkpoint_dir = os.path.join(os.getcwd(), 'ode')
        if not os.path.exists(self.checkpoint_dir):
            self.filesystem.create_dir(self.checkpoint_dir)
        self.timestamp = None

        # create model
        self.model = self.create()

        # loss function
        self.loss_fn = None

    def init_train(self, datasets=None, fold=None, niters=None, valfreq=None, lr=None):
        # dict: 'training', 'validation', 'test'
        self.datasets = datasets
        self.fold = fold
        self.niters = niters
        self.val_freq = valfreq
        self.learning_rate = lr

        # data generator
        params_train = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': self.num_workers}
        params_test = {'batch_size': 1, 'shuffle': True, 'num_workers': self.num_workers}
        self.generators = {'training': data.DataLoader(self.datasets['training'], **params_train),
                           'validation': data.DataLoader(self.datasets['validation'], **params_test),
                           'testing': data.DataLoader(self.datasets['testing'], **params_test),}

        # visualisation
        # self.fig, self.ax = self.viz_pred_init()

    def init_test(self, datasets=None):
        # dict: 'test'
        self.datasets = datasets

        # data generator
        params_test = {'batch_size': 1, 'shuffle': True, 'num_workers': self.num_workers}
        self.generators = {'testing': data.DataLoader(self.datasets['testing'], **params_test)}

    def save_state(self, itr=None):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'f'+str(self.fold)+'-i'+str(itr)+'.pth'))

    def load_state(self, timestamp=None, filename=None):
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, timestamp+'/'+filename)))
        self.model.eval()

        print('Model state_dict loaded:')
        # for param_tensor in self.model.state_dict():
        #     print('     {}   {}'.format(param_tensor, self.model.state_dict()[param_tensor].size()))

    def create(self):
        # model = ODENet(dof=self.dof, hidden_nodes=300).cuda()
        # model = ODENet3D(dof=self.dof, ode_hidden_nodes=200).cuda()
        model = ODENet3DFull(dof=self.dof, features_hidden_nodes=3, ode_hidden_nodes=100).cuda() # 100, 200
        return model

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def loss_function(self, pred, impl, plan):
        # Mean of absolute differences
        # loss = torch.mean(torch.abs(output-target))

        # MSE loss
        # loss = torch.mean((pred-impl)**2)

        # Penalised sum of component losses
        # loss = torch.mean((output[0]-target[0])**2)
        # loss += torch.mean((output[1]-target[1])**2)
        # loss += torch.mean((output[2]-target[2])**2)

        # penalise with condition:
        # https://discuss.pytorch.org/t/loss-to-penalize-overestimation/60330
        # penalizes (y_pred - y_true) > 0 approximately twice as much as the reverse
        # diff = torch.linspace(-5, 5, 11)
        # diffSqr = diff ** 2
        # mask = torch.sigmoid(diff)
        # losses = diffSqr * (1 - mask) + diffSqr * mask * 2
        # loss = losses.mean()

        # Example to penalise electrode based on impl-plan-pred
        # penalise more if pred has different direction or overestimates prediction:
        #   wrong direction: diff_p/diff_i same sign but opposite sign to diff_b
        #   underestimate: diff_p and diff_i have opposite signs
        #   overestimate: diff_p/diff_i/diff_b have same sign
        # N = 17
        # plan = torch.tensor([-0.5])
        # impl = torch.tensor([0.5])
        # pred = torch.linspace(-2.0, 2.0, N)    # at 0.25 intervals
        # diff_b = torch.ones(pred.size())*(impl-plan)

        diff_b = impl-plan    # displacement due to bending
        diff_p = pred-plan
        diff_i = pred-impl
        sign_p = torch.sign(diff_p)
        sign_i = torch.sign(diff_i)
        sign_b = torch.sign(diff_b)
        diffSqr_p = diff_p ** 2
        diffSqr_i = diff_i ** 2
        mask_p = torch.sigmoid(diff_p)
        mask_i = torch.sigmoid(diff_i)
        losses_p = 3*diffSqr_p * (1 - mask_p) + diffSqr_p * mask_p
        losses_i = diffSqr_i * (1 - mask_i) + 2*diffSqr_i * mask_i

        # but losses_p should only include wrong direction
        # i.e. diff_p/diff_i have same sign but opposite sign to diff_b
        cond_diff_same_sign = (sign_p * sign_i) == 1
        cond_bend_opp_sign = (-sign_p) == sign_b
        losses_p_zero = torch.where(cond_diff_same_sign & cond_bend_opp_sign, losses_p, torch.tensor([0.0], dtype=torch.float64).to(self.device))
        losses = losses_p_zero + losses_i
        loss = losses.mean()

        return loss

    def train(self):
        # odeint type
        if self.adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        # create timestamp dir in checkpoint
        if self.timestamp is None:
            now = datetime.now()
            self.timestamp = now.strftime('%Y%m%d-%H%M')
            self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.timestamp)
            self.filesystem.create_dir(self.checkpoint_dir)
        else:
            self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.timestamp)

        # save data
        # self.datasets['training'].save(checkpoint_dir=self.checkpoint_dir, fold=self.fold, type='training')
        # self.datasets['validation'].save(checkpoint_dir=self.checkpoint_dir, fold=self.fold, type='validation')
        # self.datasets['testing'].save(checkpoint_dir=self.checkpoint_dir, fold=self.fold, type='testing')

        print('Number of parameters: {}'.format(self.count_parameters()))
        print('Model:\n', self.model)

        self.loss_fn = torch.nn.MSELoss()

        # optimizer = optim.RMSprop(self.model.parameters(), lr=1e-3)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, cooldown=10, verbose=True)
        print('Optimiser state_dict:')
        for var_name in optimizer.state_dict():
            print('     {}      {}'.format(var_name, optimizer.state_dict()[var_name]))

        end = time.time()

        time_meter = RunningAverageMeter(0.97)
        loss_meter = RunningAverageMeter(0.97)

        for epoch in range(1, self.niters + 1):
            self.model.train()
            optimizer.zero_grad()

            # scheduler (learning step)
            for param_group in optimizer.param_groups:
                self.learning_rate = param_group['lr']

            # get batches
            train_epoch_loss = []
            train_epoch_loss_mse = []
            train_epoch_loss_x, train_epoch_loss_y, train_epoch_loss_z, train_epoch_loss_w = [], [], [], []
            for t, batch_y0, batch_y, plan, ep, dir, case, name in self.generators['training']:
                t = t[0, :]
                # print('training: t[{}] = {}'.format(t.shape, t))
                batch_y = batch_y.permute([1, 0, 2, 3])
                # print('training: t={} y0={} y={} case={} name={}'.format(t.shape, batch_y0.shape, batch_y.shape, case, name))

                t = t.to(self.device)
                batch_y0 = batch_y0.to(self.device)
                batch_y = batch_y.to(self.device)
                ep = ep.to(self.device)
                dir = dir.to(self.device)
                # window = window.to(self.device)
                window = torch.tensor([0.0], dtype=torch.float32).to(self.device)

                pred_y = odeint(self.model, batch_y0, t)
                '''  Change  '''
                #(pred_y, _, _, _) = odeint(self.model, (batch_y0, ep, dir, window), t)

                ''' loss function '''
                # MSE
                loss_mse = self.loss_fn(pred_y, batch_y)
                loss = loss_mse

                # all fields
                # loss = torch.mean(torch.abs(pred_y - batch_y))

                # only 3D position
                # loss = torch.mean(torch.abs(pred_y[:, :, :, 0:3] - batch_y[:, :, :, 0:3]))  # only positions

                # penalisation (overestimate, underestimate, wrong) and per component
                # ''' xyz '''
                # plan = plan.to(self.device)
                # plan_0 = torch.reshape(plan[:, :, 0], (-1,))
                # plan_1 = torch.reshape(plan[:, :, 1], (-1,))
                # plan_2 = torch.reshape(plan[:, :, 2], (-1,))
                # impl_y_0 = torch.reshape(batch_y[:, :, :, 0], (-1,))
                # impl_y_1 = torch.reshape(batch_y[:, :, :, 1], (-1,))
                # impl_y_2 = torch.reshape(batch_y[:, :, :, 2], (-1,))
                # pred_y_0 = torch.reshape(pred_y[:, :, :, 0], (-1,))
                # pred_y_1 = torch.reshape(pred_y[:, :, :, 1], (-1,))
                # pred_y_2 = torch.reshape(pred_y[:, :, :, 2], (-1,))
                # loss_x = self.loss_function(pred_y_0, impl_y_0, plan_0)
                # loss_y = self.loss_function(pred_y_1, impl_y_1, plan_1)
                # loss_z = self.loss_function(pred_y_2, impl_y_2, plan_2)
                # # loss_w = torch.mean(torch.abs(pred_y[:, :, :, -6:] - batch_y[:, :, :, -6:]))
                # # loss = loss_x + 10.0*loss_y + 10.0*loss_z
                # # loss = loss_x + loss_y + loss_z + loss_w
                # loss = loss_x + loss_y + loss_z

                ''' yz '''
                # plan = plan.to(self.device)
                # plan_0 = torch.reshape(plan[:, :, 0], (-1,))
                # plan_1 = torch.reshape(plan[:, :, 1], (-1,))
                # plan_2 = torch.reshape(plan[:, :, 2], (-1,))
                # impl_y_1 = torch.reshape(batch_y[:, :, :, 0], (-1,))
                # impl_y_2 = torch.reshape(batch_y[:, :, :, 1], (-1,))
                # pred_y_1 = torch.reshape(pred_y[:, :, :, 0], (-1,))
                # pred_y_2 = torch.reshape(pred_y[:, :, :, 1], (-1,))
                # loss_y = self.loss_function(pred_y_1, impl_y_1, plan_1)
                # loss_z = self.loss_function(pred_y_2, impl_y_2, plan_2)
                # loss_w = torch.mean((pred_y[:, :, :, -6:] - batch_y[:, :, :, -6:]) ** 2)
                # loss = loss_y + loss_z + loss_w
                # print('loss={} loss_y={} loss_z={} loss_w={}'.format(loss, loss_y, loss_z, loss_w))

                loss.backward()
                optimizer.step()

                time_meter.update(time.time() - end)
                loss_meter.update(loss.item())
                train_epoch_loss.append(loss.item())
                train_epoch_loss_mse.append(loss_mse.item())
                # train_epoch_loss_x.append(loss_x.item())
                # train_epoch_loss_y.append(loss_y.item())
                # train_epoch_loss_z.append(loss_z.item())
                # train_epoch_loss_w.append(loss_w.item())

            train_epoch_loss = np.mean(train_epoch_loss)
            train_epoch_loss_mse = np.mean(train_epoch_loss_mse)
            # train_epoch_loss_x = np.mean(train_epoch_loss_x)
            # train_epoch_loss_y = np.mean(train_epoch_loss_y)
            # train_epoch_loss_z = np.mean(train_epoch_loss_z)
            # train_epoch_loss_w = np.mean(train_epoch_loss_w)
            print('Training: Fold {:04d} | Iter {:04d} | lr {:06f} | Total Loss {:.6f} (mse={:.6f})'.format(self.fold, epoch, self.learning_rate, train_epoch_loss, train_epoch_loss_mse))
            # print('Training: Fold {:04d} | Iter {:04d} | Total Loss {:.6f} (x={:.6f}, y={:.6f}, z={:.6f})'.format(self.fold, epoch, train_epoch_loss, train_epoch_loss_x, train_epoch_loss_y, train_epoch_loss_z))
            if epoch == 1:
                self.best_val_loss = train_epoch_loss
                self.best_val_itr = epoch
                self.save_state(itr=epoch)

            if epoch % self.val_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    val_epoch_loss = []
                    val_epoch_loss_mse = []
                    val_epoch_loss_x, val_epoch_loss_y, val_epoch_loss_z, val_epoch_loss_w = [], [], [], []
                    for t, batch_y0, batch_y, plan, ep, dir, case, name in self.generators['validation']:
                        t = t[0, :]
                        batch_y = batch_y.permute([1, 0, 2, 3])
                        # print('validation t={} y0={} y={} case={} name={}'.format(t.shape, batch_y0.shape, batch_y.shape, case, name))

                        t = t.to(self.device)
                        batch_y0 = batch_y0.to(self.device)
                        batch_y = batch_y.to(self.device)
                        ep = ep.to(self.device)
                        dir = dir.to(self.device)
                        window = torch.tensor([0.0], dtype=torch.float32).to(self.device)

                        # pred_y = odeint(self.model, batch_y0, t)
                        (pred_y, _, _, _)  = odeint(self.model, (batch_y0, ep, dir, window), t)

                        # MSE
                        loss_mse = self.loss_fn(pred_y, batch_y)
                        loss = loss_mse

                        # loss = torch.mean(torch.abs(pred_y - batch_y))
                        # loss = torch.mean(torch.abs(pred_y[:, :, :, 0:3] - batch_y[:, :, :, 0:3]))
                        # penalisation (overestimate, underestimate, wrong) and per component
                        # ''' xyz '''
                        # plan = plan.to(self.device)
                        # plan_0 = torch.reshape(plan[:, :, 0], (-1,))
                        # plan_1 = torch.reshape(plan[:, :, 1], (-1,))
                        # plan_2 = torch.reshape(plan[:, :, 2], (-1,))
                        # impl_y_0 = torch.reshape(batch_y[:, :, :, 0], (-1,))
                        # impl_y_1 = torch.reshape(batch_y[:, :, :, 1], (-1,))
                        # impl_y_2 = torch.reshape(batch_y[:, :, :, 2], (-1,))
                        # pred_y_0 = torch.reshape(pred_y[:, :, :, 0], (-1,))
                        # pred_y_1 = torch.reshape(pred_y[:, :, :, 1], (-1,))
                        # pred_y_2 = torch.reshape(pred_y[:, :, :, 2], (-1,))
                        # loss_x = self.loss_function(pred_y_0, impl_y_0, plan_0)
                        # loss_y = self.loss_function(pred_y_1, impl_y_1, plan_1)
                        # loss_z = self.loss_function(pred_y_2, impl_y_2, plan_2)
                        # # loss_w = torch.mean(torch.abs(pred_y[:, :, :, -6:] - batch_y[:, :, :, -6:]))
                        # # loss = loss_x + 10.0*loss_y + 10.0*loss_z
                        # # loss = loss_x + loss_y + loss_z + loss_w
                        # loss = loss_x + loss_y + loss_z

                        ''' yz '''
                        # plan = plan.to(self.device)
                        # plan_0 = torch.reshape(plan[:, :, 0], (-1,))
                        # plan_1 = torch.reshape(plan[:, :, 1], (-1,))
                        # plan_2 = torch.reshape(plan[:, :, 2], (-1,))
                        # impl_y_1 = torch.reshape(batch_y[:, :, :, 0], (-1,))
                        # impl_y_2 = torch.reshape(batch_y[:, :, :, 1], (-1,))
                        # pred_y_1 = torch.reshape(pred_y[:, :, :, 0], (-1,))
                        # pred_y_2 = torch.reshape(pred_y[:, :, :, 1], (-1,))
                        # loss_y = self.loss_function(pred_y_1, impl_y_1, plan_1)
                        # loss_z = self.loss_function(pred_y_2, impl_y_2, plan_2)
                        # loss_w = torch.mean((pred_y[:, :, :, -6:] - batch_y[:, :, :, -6:]) ** 2)
                        # loss = loss_y + loss_z + loss_w

                        val_epoch_loss.append(loss.item())
                        val_epoch_loss_mse.append(loss_mse.item())
                        # val_epoch_loss_x.append(loss_x.item())
                        # val_epoch_loss_y.append(loss_y.item())
                        # val_epoch_loss_z.append(loss_z.item())
                        # val_epoch_loss_w.append(loss_w.item())

                    # save model if there is improvement
                    val_epoch_loss = np.mean(val_epoch_loss)
                    val_epoch_loss_mse = np.mean(val_epoch_loss_mse)
                    # val_epoch_loss_x = np.mean(val_epoch_loss_x)
                    # val_epoch_loss_y = np.mean(val_epoch_loss_y)
                    # val_epoch_loss_z = np.mean(val_epoch_loss_z)
                    # val_epoch_loss_w = np.mean(val_epoch_loss_w)
                    print('Validation: Fold {:04d} | Iter {:04d} | Total Loss {:.6f} (mse={:.6f})'.format(self.fold, epoch, val_epoch_loss, val_epoch_loss_mse))
                    # print('Validation: Fold {:04d} | Iter {:04d} | Total Loss {:.6f} (x={:.6f}, y={:.6f}, z={:.6f})'.format(self.fold, epoch, val_epoch_loss, val_epoch_loss_x, val_epoch_loss_y, val_epoch_loss_z))
                    if val_epoch_loss < self.best_val_loss:
                        self.best_val_loss = val_epoch_loss
                        self.best_val_itr = epoch
                        self.save_state(itr=epoch)
                        print('     [INFO] checkpoint with val_epoch_loss={} saved!'.format(val_epoch_loss))

                    # visualisation
                    # self.viz_pred_clear()
                    # for i in range(len(self.datasets['training'].dataset['case'])):
                    #     pred_y = odeint(self.model, self.datasets['training'].dataset['true_y0'][i].to(self.device),
                    #                     self.datasets['training'].dataset['t'][i].to(self.device))
                    #     self.viz_pred_plot(true=self.datasets['training'].dataset['true_y'][i], pred=pred_y.cpu())
                    # self.viz_pred_draw()

            # scheduler
            # scheduler.step()    # StepLR
            scheduler.step(train_epoch_loss)  # ReduceLROnPlateau

        end = time.time()

    def test(self):
        # odeint type
        if self.adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        self.model.eval()
        with torch.no_grad():
            self.test_loss = []
            test_loss_mse = []
            test_loss_x, test_loss_y, test_loss_z, test_loss_w = [], [], [], []
            for t, batch_y0, batch_y, plan, ep, dir, case, name in self.generators['testing']:
                t = t[0, :]
                batch_y = batch_y.permute([1, 0, 2, 3])
                # print('testing t={} y0={} y={} case={} name={}'.format(t.shape, batch_y0.shape, batch_y.shape, case, name))

                t = t.to(self.device)
                batch_y0 = batch_y0.to(self.device)
                batch_y = batch_y.to(self.device)
                ep = ep.to(self.device)
                dir = dir.to(self.device)
                window = torch.tensor([0.0], dtype=torch.float32).to(self.device)

                # pred_y = odeint(self.model, batch_y0, t)
                (pred_y, _, _, _) = odeint(self.model, (batch_y0, ep, dir, window), t)

                # MSE
                loss_mse = self.loss_fn(pred_y, batch_y)
                loss = loss_mse

                # loss = torch.mean(torch.abs(pred_y - batch_y))
                # loss = torch.mean(torch.abs(pred_y[:, :, :, 0:3] - batch_y[:, :, :, 0:3]))
                # penalisation (overestimate, underestimate, wrong) and per component
                # ''' xyz '''
                # plan = plan.to(self.device)
                # plan_0 = torch.reshape(plan[:, :, 0], (-1,))
                # plan_1 = torch.reshape(plan[:, :, 1], (-1,))
                # plan_2 = torch.reshape(plan[:, :, 2], (-1,))
                # impl_y_0 = torch.reshape(batch_y[:, :, :, 0], (-1,))
                # impl_y_1 = torch.reshape(batch_y[:, :, :, 1], (-1,))
                # impl_y_2 = torch.reshape(batch_y[:, :, :, 2], (-1,))
                # pred_y_0 = torch.reshape(pred_y[:, :, :, 0], (-1,))
                # pred_y_1 = torch.reshape(pred_y[:, :, :, 1], (-1,))
                # pred_y_2 = torch.reshape(pred_y[:, :, :, 2], (-1,))
                # loss_x = self.loss_function(pred_y_0, impl_y_0, plan_0)
                # loss_y = self.loss_function(pred_y_1, impl_y_1, plan_1)
                # loss_z = self.loss_function(pred_y_2, impl_y_2, plan_2)
                # # loss_w = torch.mean(torch.abs(pred_y[:, :, :, -6:] - batch_y[:, :, :, -6:]))
                # # loss = loss_x + 10.0*loss_y + 10.0*loss_z
                # # loss = loss_x + loss_y + loss_z + loss_w
                # loss = loss_x + loss_y + loss_z

                ''' yz '''
                # plan = plan.to(self.device)
                # plan_0 = torch.reshape(plan[:, :, 0], (-1,))
                # plan_1 = torch.reshape(plan[:, :, 1], (-1,))
                # plan_2 = torch.reshape(plan[:, :, 2], (-1,))
                # impl_y_1 = torch.reshape(batch_y[:, :, :, 0], (-1,))
                # impl_y_2 = torch.reshape(batch_y[:, :, :, 1], (-1,))
                # pred_y_1 = torch.reshape(pred_y[:, :, :, 0], (-1,))
                # pred_y_2 = torch.reshape(pred_y[:, :, :, 1], (-1,))
                # loss_y = self.loss_function(pred_y_1, impl_y_1, plan_1)
                # loss_z = self.loss_function(pred_y_2, impl_y_2, plan_2)
                # loss_w = torch.mean((pred_y[:, :, :, -6:] - batch_y[:, :, :, -6:]) ** 2)
                # loss = loss_y + loss_z + loss_w

                self.test_loss.append(loss.item())
                test_loss_mse.append(loss_mse.item())
                # test_loss_x.append(loss_x.item())
                # test_loss_y.append(loss_y.item())
                # test_loss_z.append(loss_z.item())
                # test_loss_w.append(loss_w.item())

            self.test_loss = np.mean(self.test_loss)
            test_loss_mse = np.mean(test_loss_mse)
            # test_loss_x = np.mean(test_loss_x)
            # test_loss_y = np.mean(test_loss_y)
            # test_loss_z = np.mean(test_loss_z)
            # test_loss_w = np.mean(test_loss_w)
            print('Test Loss = {:.6f} (mse={:.6f})'.format(self.test_loss, test_loss_mse))
            # print('Test Loss = {:.6f} (x={:.6f}, y={:.6f}, z={:.6f})'.format(self.test_loss, test_loss_x, test_loss_y, test_loss_z))

    def infer(self):
        # odeint type
        if self.adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint

        pred_dict = {'case': [], 'name': [], 'points': []}
        self.model.eval()
        with torch.no_grad():
            for t, batch_y0, batch_y, plan, ep, dir, case, name in self.generators['testing']:
                t = t[0, :]
                batch_y = batch_y.permute([1, 0, 2, 3])
                print('inference t={} y0={} y={} case={} name={}'.format(t.shape, batch_y0.shape, batch_y.shape, case, name))

                t = t.to(self.device)
                batch_y0 = batch_y0.to(self.device)
                batch_y = batch_y.to(self.device)
                ep = ep.to(self.device)
                dir = dir.to(self.device)
                window = torch.zeros((9, 9, 9), dtype=torch.float32).to(self.device)

                # pred_y = odeint(self.model, batch_y0, t)
                (pred_y, _, _, _)  = odeint(self.model, (batch_y0, ep, dir, window), t)

                points = pred_y[:, :, :, 0:3]
                points = torch.reshape(points, (points.size()[0], points.size()[-1]))

                # save
                pred_dict['case'].append(case)
                pred_dict['name'].append(name)
                pred_dict['points'].append(points)

        return pred_dict

    def viz_pred_init(self):
        ax = [0,0,0]
        if self.viz:
            fig = plt.figure(figsize=(12, 4), facecolor='white')
            ax[0] = fig.add_subplot(131, frameon=False)
            ax[1] = fig.add_subplot(132, frameon=False)
            ax[2] = fig.add_subplot(133, frameon=False)
            plt.show(block=False)
        return fig, ax

    def viz_pred_clear(self):
        if self.viz:
            self.ax[0].cla()
            self.ax[0].set_title('Trajectories')
            self.ax[0].set_xlabel('t')
            self.ax[0].set_ylabel('x')
            self.ax[0].set_xlim(0, 80)
            self.ax[0].set_ylim(-80, 80)

            self.ax[1].cla()
            self.ax[1].set_title('Trajectories')
            self.ax[1].set_xlabel('t')
            self.ax[1].set_ylabel('y')
            self.ax[1].set_xlim(0, 80)
            self.ax[1].set_ylim(-75, 110)

            self.ax[2].cla()
            self.ax[2].set_title('Trajectories')
            self.ax[2].set_xlabel('t')
            self.ax[2].set_ylabel('z')
            self.ax[2].set_xlim(0, 80)
            self.ax[2].set_ylim(-60, 90)

    def viz_pred_plot(self, true=None, pred=None):
        if self.viz:
            depth = np.arange(len(true))

            self.ax[0].plot(depth, true.numpy()[:, 0, 0], 'r-')
            self.ax[0].plot(depth, pred.numpy()[:, 0, 0], 'r--')

            self.ax[1].plot(depth, true.numpy()[:, 0, 1], 'g-')
            self.ax[1].plot(depth, pred.numpy()[:, 0, 1], 'g--')

            self.ax[2].plot(depth, true.numpy()[:, 0, 2], 'b-')
            self.ax[2].plot(depth, pred.numpy()[:, 0, 2], 'b--')

    def viz_pred_draw(self):
        if self.viz:
            self.fig.tight_layout()
            plt.draw()
            plt.pause(self.delay)