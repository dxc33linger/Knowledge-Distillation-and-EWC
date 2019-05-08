"""
Author: XD
Date: April 2019
Project: Continual learning with EWC KD

"""

import logging
import os
import random
import re
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import pickle
from resnet32_progressive import *
from resnet32_without_bridge import *
from utils_tool import progress_bar
from vgg16_progressive import *
from vgg16_without_bridge import *
from resnet32_B import *
args = parser.parse_args()


class ContinualNN(object):
    def __init__(self):
        self.batch_size = args.batch_size
 
    def train_with_mask_with_EWC(self, current_trainloader, previous_trainloader,  fisher_estimation_sample_size=256): # retrain percentage
        all_loader = []
        all_loader.append(previous_trainloader)
        all_loader.append(current_trainloader)

        for task_id, trainloader in enumerate(all_loader):
            # self.initialization(args.lr_mutant, args.lr_mutant_step_size, args.weight_decay_2)
            self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr_mutant, momentum=0.9, weight_decay=args.weight_decay_2)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_mutant_step_size, gamma=args.lr_gamma)
            # self.save_mutant(55, 1)
            train_acc = np.zeros([1, args.num_epoch])
            test_acc = np.zeros([1, args.num_epoch])

            for epoch in range(args.num_epoch):

                lr_list = self.scheduler.get_lr()
                logging.info('\nEpoch: %d lr: %s' % (epoch, self.scheduler.get_lr()))
                self.scheduler.step()
                self.net.train()
                train_loss = 0.0
                correct = 0
                total = 0

                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    # print(batch_idx)
                    # targets =  self.make_one_hot(targets, len_onehot)
                    # print(target)
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    inputs_var = Variable(inputs)
                    targets_var = Variable(targets)

                    self.optimizer.zero_grad()
                    outputs = self.net(inputs_var)
                    loss = self.criterion(outputs, targets_var)

                    if args.ewc and task_id > 0 :
                        ewc_loss = self.net.ewc_loss(cuda = self.device)
                    else:
                        ewc_loss = 0.0

                    loss = loss + ewc_loss
                    loss.backward()
                    self.optimizer.step()

                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    self.loss = loss
                   

                progress_bar(batch_idx, len(trainloader), ' | Loss:%.3f| ewc_loss:%.3f | Acc:%.3f%% (%d/%d) -- Train Task(%d/%d)'
                     % ( loss, ewc_loss, 100.*correct/total, correct, total, task_id, len(all_loader)))
                train_acc[0, epoch] = correct / total
                # test_acc[0, epoch] = self.test(trainloader)


            if args.consolidate and task_id < len(all_loader):
                # estimate the fisher information of the parameters and consolidate
                # them in the network.
                print(
                    '=> Estimating diagonals of the fisher information matrix...',
                    end='', flush=True
                )
                self.net.consolidate(self.net.estimate_fisher(
                    trainloader, fisher_estimation_sample_size
                ))
                print(' Done!')
            else:
                logging.info('No consolidate/EWC loss available')

        return test_acc
        # return correct / total



    def test(self, testloader):
        self.net.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = Variable(inputs)
                targets = Variable(targets)

                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Test'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            #
            # print('target', targets)
            # print('predicted', predicted)
        return correct/total


    def make_one_hot(self, labels, C):
        # Converts an integer label torch.autograd.Variable to a one-hot Variable.
        labels = torch.unsqueeze(labels, 1)
        one_hot = torch.zeros(args.batch_size, C).scatter_(1, torch.LongTensor(labels), 1)
        target = Variable(one_hot)
        # print('one hot', target)
        return target


    def train_with_mask_with_KD(self, epoch, trainloader, KD_target_list, len_onehot):
        mask_dict = pickle.load(open(save_mask_file, "rb"))
        mask_reverse_dict = pickle.load(open(save_mask_fileR, "rb"))

        lr_list = self.scheduler.get_lr()
        logging.info('\nEpoch: %d lr: %s' % (epoch, self.scheduler.get_lr()))
        self.scheduler.step()
        self.net.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            targets_KD =  self.make_one_hot(targets, len_onehot)
            w = KD_target_list[batch_idx].shape[1]
            targets_KD[:, 0:w] = KD_target_list[batch_idx][:, 0:w]

            inputs, targets_KD, targets = inputs.to(self.device), targets_KD.to(self.device), targets.to(self.device)

            inputs_var = Variable(inputs)
            targets_var_KD = Variable(targets_KD)

            self.optimizer.zero_grad()
            # print(targets_var_KD.shape)
            outputs = self.net(inputs_var)/args.temperature
            loss = self.xentropy_cost(outputs, targets_var_KD)
            # break
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets_var_KD.size(0)
            correct += predicted.eq(targets).sum().item()
            self.loss = train_loss

            progress_bar(batch_idx, len(trainloader), 'Loss:%.3f|Acc:%.3f%% (%d/%d)--Train' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return correct / total


    def obtain_KD_target(self, testloader):
        self.net.eval()

        KD_target_list = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = Variable(inputs)

                outputs = self.net(inputs)/args.temperature
                KD_target = F.softmax(outputs, dim=1)
                KD_target_list.append(KD_target)
        return KD_target_list



    def xentropy_cost(self, input, target):  #Cross entropy that accepts soft targets
        sftmx = nn.Softmax()
        loss = torch.mean(torch.sum(-target* torch.log(sftmx(input)), dim=1))
        # print(sftmx(input)[0:2, :])
        return loss
        # if size_average:
        #     return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
        # else:
        #     return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

