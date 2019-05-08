import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch import autograd
from args import parser
args = parser.parse_args()

class ResNet32_AB_noBridge(nn.Module):
   def __init__(self, task_id, num_classes, init_weights=True):
       super(ResNet32_AB_noBridge, self).__init__()
       self.task_id = task_id
       self.num_classes_A = num_classes*1
       self.num_classes_B = num_classes*2

       self.conv_0_0_A = nn.Conv2d(3, args.NA_C0, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn_0_0_A = nn.BatchNorm2d(args.NA_C0)

       self.shortcut_1_1_A = nn.Sequential()
       self.conv1_1_1_A = nn.Conv2d(args.NA_C0*1, args.NA_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_1_1_A = nn.BatchNorm2d(args.NA_C0*1)
       self.conv2_1_1_A = nn.Conv2d(args.NA_C0*1, args.NA_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_1_1_A = nn.BatchNorm2d(args.NA_C0*1)

       self.shortcut_1_2_A = nn.Sequential()
       self.conv1_1_2_A = nn.Conv2d(args.NA_C0*1, args.NA_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_1_2_A = nn.BatchNorm2d(args.NA_C0*1)
       self.conv2_1_2_A = nn.Conv2d(args.NA_C0*1, args.NA_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_1_2_A = nn.BatchNorm2d(args.NA_C0*1)

       self.shortcut_1_3_A = nn.Sequential()
       self.conv1_1_3_A = nn.Conv2d(args.NA_C0*1, args.NA_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_1_3_A = nn.BatchNorm2d(args.NA_C0*1)
       self.conv2_1_3_A = nn.Conv2d(args.NA_C0*1, args.NA_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_1_3_A = nn.BatchNorm2d(args.NA_C0*1)

       self.shortcut_1_4_A = nn.Sequential()
       self.conv1_1_4_A = nn.Conv2d(args.NA_C0*1, args.NA_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_1_4_A = nn.BatchNorm2d(args.NA_C0*1)
       self.conv2_1_4_A = nn.Conv2d(args.NA_C0*1, args.NA_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_1_4_A = nn.BatchNorm2d(args.NA_C0*1)

       self.shortcut_1_5_A = nn.Sequential()
       self.conv1_1_5_A = nn.Conv2d(args.NA_C0*1, args.NA_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_1_5_A = nn.BatchNorm2d(args.NA_C0*1)
       self.conv2_1_5_A = nn.Conv2d(args.NA_C0*1, args.NA_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_1_5_A = nn.BatchNorm2d(args.NA_C0*1)

       self.shortcut_2_1_A = nn.Sequential(
           nn.Conv2d(args.NA_C0*1, args.NA_C0*2, kernel_size=1, stride=2, bias=False),
           nn.BatchNorm2d(args.NA_C0*2) )
       self.conv1_2_1_A = nn.Conv2d(args.NA_C0*1, args.NA_C0*2, kernel_size=3, stride=2, padding=1, bias=False)
       self.bn1_2_1_A = nn.BatchNorm2d(args.NA_C0*2)
       self.conv2_2_1_A = nn.Conv2d(args.NA_C0*2, args.NA_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_2_1_A = nn.BatchNorm2d(args.NA_C0*2)

       self.shortcut_2_2_A = nn.Sequential()
       self.conv1_2_2_A = nn.Conv2d(args.NA_C0*2, args.NA_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_2_2_A = nn.BatchNorm2d(args.NA_C0*2)
       self.conv2_2_2_A = nn.Conv2d(args.NA_C0*2, args.NA_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_2_2_A = nn.BatchNorm2d(args.NA_C0*2)

       self.shortcut_2_3_A = nn.Sequential()
       self.conv1_2_3_A = nn.Conv2d(args.NA_C0*2, args.NA_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_2_3_A = nn.BatchNorm2d(args.NA_C0*2)
       self.conv2_2_3_A = nn.Conv2d(args.NA_C0*2, args.NA_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_2_3_A = nn.BatchNorm2d(args.NA_C0*2)

       self.shortcut_2_4_A = nn.Sequential()
       self.conv1_2_4_A = nn.Conv2d(args.NA_C0*2, args.NA_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_2_4_A = nn.BatchNorm2d(args.NA_C0*2)
       self.conv2_2_4_A = nn.Conv2d(args.NA_C0*2, args.NA_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_2_4_A = nn.BatchNorm2d(args.NA_C0*2)

       self.shortcut_2_5_A = nn.Sequential()
       self.conv1_2_5_A = nn.Conv2d(args.NA_C0*2, args.NA_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_2_5_A = nn.BatchNorm2d(args.NA_C0*2)
       self.conv2_2_5_A = nn.Conv2d(args.NA_C0*2, args.NA_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_2_5_A = nn.BatchNorm2d(args.NA_C0*2)

       self.shortcut_3_1_A = nn.Sequential(
           nn.Conv2d(args.NA_C0*2, args.NA_C0*4, kernel_size=1, stride=2, bias=False),
           nn.BatchNorm2d(args.NA_C0*4) )
       self.conv1_3_1_A = nn.Conv2d(args.NA_C0*2, args.NA_C0*4, kernel_size=3, stride=2, padding=1, bias=False)
       self.bn1_3_1_A = nn.BatchNorm2d(args.NA_C0*4)
       self.conv2_3_1_A = nn.Conv2d(args.NA_C0*4, args.NA_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_3_1_A = nn.BatchNorm2d(args.NA_C0*4)

       self.shortcut_3_2_A = nn.Sequential()
       self.conv1_3_2_A = nn.Conv2d(args.NA_C0*4, args.NA_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_3_2_A = nn.BatchNorm2d(args.NA_C0*4)
       self.conv2_3_2_A = nn.Conv2d(args.NA_C0*4, args.NA_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_3_2_A = nn.BatchNorm2d(args.NA_C0*4)

       self.shortcut_3_3_A = nn.Sequential()
       self.conv1_3_3_A = nn.Conv2d(args.NA_C0*4, args.NA_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_3_3_A = nn.BatchNorm2d(args.NA_C0*4)
       self.conv2_3_3_A = nn.Conv2d(args.NA_C0*4, args.NA_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_3_3_A = nn.BatchNorm2d(args.NA_C0*4)

       self.shortcut_3_4_A = nn.Sequential()
       self.conv1_3_4_A = nn.Conv2d(args.NA_C0*4, args.NA_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_3_4_A = nn.BatchNorm2d(args.NA_C0*4)
       self.conv2_3_4_A = nn.Conv2d(args.NA_C0*4, args.NA_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_3_4_A = nn.BatchNorm2d(args.NA_C0*4)

       self.shortcut_3_5_A = nn.Sequential()
       self.conv1_3_5_A = nn.Conv2d(args.NA_C0*4, args.NA_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_3_5_A = nn.BatchNorm2d(args.NA_C0*4)
       self.conv2_3_5_A = nn.Conv2d(args.NA_C0*4, args.NA_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_3_5_A = nn.BatchNorm2d(args.NA_C0*4)

       self.conv_0_0_B = nn.Conv2d(3, args.NB_C0, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn_0_0_B = nn.BatchNorm2d(args.NB_C0)

       self.shortcut_1_1_B = nn.Sequential()
       self.conv1_1_1_B = nn.Conv2d(args.NB_C0*1, args.NB_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_1_1_B = nn.BatchNorm2d(args.NB_C0*1)
       self.conv2_1_1_B = nn.Conv2d(args.NB_C0*1, args.NB_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_1_1_B = nn.BatchNorm2d(args.NB_C0*1)

       self.shortcut_1_2_B = nn.Sequential()
       self.conv1_1_2_B = nn.Conv2d(args.NB_C0*1, args.NB_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_1_2_B = nn.BatchNorm2d(args.NB_C0*1)
       self.conv2_1_2_B = nn.Conv2d(args.NB_C0*1, args.NB_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_1_2_B = nn.BatchNorm2d(args.NB_C0*1)

       self.shortcut_1_3_B = nn.Sequential()
       self.conv1_1_3_B = nn.Conv2d(args.NB_C0*1, args.NB_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_1_3_B = nn.BatchNorm2d(args.NB_C0*1)
       self.conv2_1_3_B = nn.Conv2d(args.NB_C0*1, args.NB_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_1_3_B = nn.BatchNorm2d(args.NB_C0*1)

       self.shortcut_1_4_B = nn.Sequential()
       self.conv1_1_4_B = nn.Conv2d(args.NB_C0*1, args.NB_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_1_4_B = nn.BatchNorm2d(args.NB_C0*1)
       self.conv2_1_4_B = nn.Conv2d(args.NB_C0*1, args.NB_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_1_4_B = nn.BatchNorm2d(args.NB_C0*1)

       self.shortcut_1_5_B = nn.Sequential()
       self.conv1_1_5_B = nn.Conv2d(args.NB_C0*1, args.NB_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_1_5_B = nn.BatchNorm2d(args.NB_C0*1)
       self.conv2_1_5_B = nn.Conv2d(args.NB_C0*1, args.NB_C0*1, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_1_5_B = nn.BatchNorm2d(args.NB_C0*1)

       self.shortcut_2_1_B = nn.Sequential(
           nn.Conv2d(args.NB_C0*1, args.NB_C0*2, kernel_size=1, stride=2, bias=False),
           nn.BatchNorm2d(args.NB_C0*2) )
       self.conv1_2_1_B = nn.Conv2d(args.NB_C0*1, args.NB_C0*2, kernel_size=3, stride=2, padding=1, bias=False)
       self.bn1_2_1_B = nn.BatchNorm2d(args.NB_C0*2)
       self.conv2_2_1_B = nn.Conv2d(args.NB_C0*2, args.NB_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_2_1_B = nn.BatchNorm2d(args.NB_C0*2)

       self.shortcut_2_2_B = nn.Sequential()
       self.conv1_2_2_B = nn.Conv2d(args.NB_C0*2, args.NB_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_2_2_B = nn.BatchNorm2d(args.NB_C0*2)
       self.conv2_2_2_B = nn.Conv2d(args.NB_C0*2, args.NB_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_2_2_B = nn.BatchNorm2d(args.NB_C0*2)

       self.shortcut_2_3_B = nn.Sequential()
       self.conv1_2_3_B = nn.Conv2d(args.NB_C0*2, args.NB_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_2_3_B = nn.BatchNorm2d(args.NB_C0*2)
       self.conv2_2_3_B = nn.Conv2d(args.NB_C0*2, args.NB_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_2_3_B = nn.BatchNorm2d(args.NB_C0*2)

       self.shortcut_2_4_B = nn.Sequential()
       self.conv1_2_4_B = nn.Conv2d(args.NB_C0*2, args.NB_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_2_4_B = nn.BatchNorm2d(args.NB_C0*2)
       self.conv2_2_4_B = nn.Conv2d(args.NB_C0*2, args.NB_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_2_4_B = nn.BatchNorm2d(args.NB_C0*2)

       self.shortcut_2_5_B = nn.Sequential()
       self.conv1_2_5_B = nn.Conv2d(args.NB_C0*2, args.NB_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_2_5_B = nn.BatchNorm2d(args.NB_C0*2)
       self.conv2_2_5_B = nn.Conv2d(args.NB_C0*2, args.NB_C0*2, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_2_5_B = nn.BatchNorm2d(args.NB_C0*2)

       self.shortcut_3_1_B = nn.Sequential(
           nn.Conv2d(args.NB_C0*2, args.NB_C0*4, kernel_size=1, stride=2, bias=False),
           nn.BatchNorm2d(args.NB_C0*4) )
       self.conv1_3_1_B = nn.Conv2d(args.NB_C0*2, args.NB_C0*4, kernel_size=3, stride=2, padding=1, bias=False)
       self.bn1_3_1_B = nn.BatchNorm2d(args.NB_C0*4)
       self.conv2_3_1_B = nn.Conv2d(args.NB_C0*4, args.NB_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_3_1_B = nn.BatchNorm2d(args.NB_C0*4)

       self.shortcut_3_2_B = nn.Sequential()
       self.conv1_3_2_B = nn.Conv2d(args.NB_C0*4, args.NB_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_3_2_B = nn.BatchNorm2d(args.NB_C0*4)
       self.conv2_3_2_B = nn.Conv2d(args.NB_C0*4, args.NB_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_3_2_B = nn.BatchNorm2d(args.NB_C0*4)

       self.shortcut_3_3_B = nn.Sequential()
       self.conv1_3_3_B = nn.Conv2d(args.NB_C0*4, args.NB_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_3_3_B = nn.BatchNorm2d(args.NB_C0*4)
       self.conv2_3_3_B = nn.Conv2d(args.NB_C0*4, args.NB_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_3_3_B = nn.BatchNorm2d(args.NB_C0*4)

       self.shortcut_3_4_B = nn.Sequential()
       self.conv1_3_4_B = nn.Conv2d(args.NB_C0*4, args.NB_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_3_4_B = nn.BatchNorm2d(args.NB_C0*4)
       self.conv2_3_4_B = nn.Conv2d(args.NB_C0*4, args.NB_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_3_4_B = nn.BatchNorm2d(args.NB_C0*4)

       self.shortcut_3_5_B = nn.Sequential()
       self.conv1_3_5_B = nn.Conv2d(args.NB_C0*4, args.NB_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1_3_5_B = nn.BatchNorm2d(args.NB_C0*4)
       self.conv2_3_5_B = nn.Conv2d(args.NB_C0*4, args.NB_C0*4, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn2_3_5_B = nn.BatchNorm2d(args.NB_C0*4)

       self.linear_A = nn.Linear((args.NA_C0+args.NB_C0)*4, self.num_classes_B)


       if init_weights:
           self._initialize_weights()


   def _initialize_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
               if m.bias is not None:
                   nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)
           elif isinstance(m, nn.Linear):
               nn.init.normal_(m.weight, 0, 0.01)
               nn.init.constant_(m.bias, 0)


   def forward(self, x):
       if self.task_id == 1:
           out_0_0_A = F.relu(self.bn_0_0_A(self.conv_0_0_A(x)))
           out_0_0_B = F.relu(self.bn_0_0_B(self.conv_0_0_B(x)))
           #out_0_0_B += out_0_0_A

           x_1_1_A = out_0_0_A
           out_1_1_A = F.relu(self.bn1_1_1_A(self.conv1_1_1_A(x_1_1_A)))
           out_1_1_A = self.bn2_1_1_A(self.conv2_1_1_A(out_1_1_A))
           out_1_1_A += self.shortcut_1_1_A(x_1_1_A)
           out_1_1_A = F.relu(out_1_1_A)

           x_1_1_B = out_0_0_B
           out_1_1_B = F.relu(self.bn1_1_1_B(self.conv1_1_1_B(x_1_1_B)))
           out_1_1_B = self.bn2_1_1_B(self.conv2_1_1_B(out_1_1_B))
           out_1_1_B += self.shortcut_1_1_B(x_1_1_B)
           out_1_1_B = F.relu(out_1_1_B)

           # out_1_1_B += out_1_1_A

           x_1_2_A = out_1_1_A
           out_1_2_A = F.relu(self.bn1_1_2_A(self.conv1_1_2_A(x_1_2_A)))
           out_1_2_A = self.bn2_1_2_A(self.conv2_1_2_A(out_1_2_A))
           out_1_2_A += self.shortcut_1_2_A(x_1_2_A)
           out_1_2_A = F.relu(out_1_2_A)

           x_1_2_B = out_1_1_B
           out_1_2_B = F.relu(self.bn1_1_2_B(self.conv1_1_2_B(x_1_2_B)))
           out_1_2_B = self.bn2_1_2_B(self.conv2_1_2_B(out_1_2_B))
           out_1_2_B += self.shortcut_1_2_B(x_1_2_B)
           out_1_2_B = F.relu(out_1_2_B)

           # out_1_2_B += out_1_2_A

           x_1_3_A = out_1_2_A
           out_1_3_A = F.relu(self.bn1_1_3_A(self.conv1_1_3_A(x_1_3_A)))
           out_1_3_A = self.bn2_1_3_A(self.conv2_1_3_A(out_1_3_A))
           out_1_3_A += self.shortcut_1_3_A(x_1_3_A)
           out_1_3_A = F.relu(out_1_3_A)

           x_1_3_B = out_1_2_B
           out_1_3_B = F.relu(self.bn1_1_3_B(self.conv1_1_3_B(x_1_3_B)))
           out_1_3_B = self.bn2_1_3_B(self.conv2_1_3_B(out_1_3_B))
           out_1_3_B += self.shortcut_1_3_B(x_1_3_B)
           out_1_3_B = F.relu(out_1_3_B)

           # out_1_3_B += out_1_3_A

           x_1_4_A = out_1_3_A
           out_1_4_A = F.relu(self.bn1_1_4_A(self.conv1_1_4_A(x_1_4_A)))
           out_1_4_A = self.bn2_1_4_A(self.conv2_1_4_A(out_1_4_A))
           out_1_4_A += self.shortcut_1_4_A(x_1_4_A)
           out_1_4_A = F.relu(out_1_4_A)

           x_1_4_B = out_1_3_B
           out_1_4_B = F.relu(self.bn1_1_4_B(self.conv1_1_4_B(x_1_4_B)))
           out_1_4_B = self.bn2_1_4_B(self.conv2_1_4_B(out_1_4_B))
           out_1_4_B += self.shortcut_1_4_B(x_1_4_B)
           out_1_4_B = F.relu(out_1_4_B)

           # out_1_4_B += out_1_4_A

           x_1_5_A = out_1_4_A
           out_1_5_A = F.relu(self.bn1_1_5_A(self.conv1_1_5_A(x_1_5_A)))
           out_1_5_A = self.bn2_1_5_A(self.conv2_1_5_A(out_1_5_A))
           out_1_5_A += self.shortcut_1_5_A(x_1_5_A)
           out_1_5_A = F.relu(out_1_5_A)

           x_1_5_B = out_1_4_B
           out_1_5_B = F.relu(self.bn1_1_5_B(self.conv1_1_5_B(x_1_5_B)))
           out_1_5_B = self.bn2_1_5_B(self.conv2_1_5_B(out_1_5_B))
           out_1_5_B += self.shortcut_1_5_B(x_1_5_B)
           out_1_5_B = F.relu(out_1_5_B)

           # out_1_5_B += out_1_5_A

           x_2_1_A = out_1_5_A
           out_2_1_A = F.relu(self.bn1_2_1_A(self.conv1_2_1_A(x_2_1_A)))
           out_2_1_A = self.bn2_2_1_A(self.conv2_2_1_A(out_2_1_A))
           out_2_1_A += self.shortcut_2_1_A(x_2_1_A)
           out_2_1_A = F.relu(out_2_1_A)

           x_2_1_B = out_1_5_B
           out_2_1_B = F.relu(self.bn1_2_1_B(self.conv1_2_1_B(x_2_1_B)))
           out_2_1_B = self.bn2_2_1_B(self.conv2_2_1_B(out_2_1_B))
           out_2_1_B += self.shortcut_2_1_B(x_2_1_B)
           out_2_1_B = F.relu(out_2_1_B)

           # out_2_1_B += out_2_1_A

           x_2_2_A = out_2_1_A
           out_2_2_A = F.relu(self.bn1_2_2_A(self.conv1_2_2_A(x_2_2_A)))
           out_2_2_A = self.bn2_2_2_A(self.conv2_2_2_A(out_2_2_A))
           out_2_2_A += self.shortcut_2_2_A(x_2_2_A)
           out_2_2_A = F.relu(out_2_2_A)

           x_2_2_B = out_2_1_B
           out_2_2_B = F.relu(self.bn1_2_2_B(self.conv1_2_2_B(x_2_2_B)))
           out_2_2_B = self.bn2_2_2_B(self.conv2_2_2_B(out_2_2_B))
           out_2_2_B += self.shortcut_2_2_B(x_2_2_B)
           out_2_2_B = F.relu(out_2_2_B)

           # out_2_2_B += out_2_2_A

           x_2_3_A = out_2_2_A
           out_2_3_A = F.relu(self.bn1_2_3_A(self.conv1_2_3_A(x_2_3_A)))
           out_2_3_A = self.bn2_2_3_A(self.conv2_2_3_A(out_2_3_A))
           out_2_3_A += self.shortcut_2_3_A(x_2_3_A)
           out_2_3_A = F.relu(out_2_3_A)

           x_2_3_B = out_2_2_B
           out_2_3_B = F.relu(self.bn1_2_3_B(self.conv1_2_3_B(x_2_3_B)))
           out_2_3_B = self.bn2_2_3_B(self.conv2_2_3_B(out_2_3_B))
           out_2_3_B += self.shortcut_2_3_B(x_2_3_B)
           out_2_3_B = F.relu(out_2_3_B)

           # out_2_3_B += out_2_3_A

           x_2_4_A = out_2_3_A
           out_2_4_A = F.relu(self.bn1_2_4_A(self.conv1_2_4_A(x_2_4_A)))
           out_2_4_A = self.bn2_2_4_A(self.conv2_2_4_A(out_2_4_A))
           out_2_4_A += self.shortcut_2_4_A(x_2_4_A)
           out_2_4_A = F.relu(out_2_4_A)

           x_2_4_B = out_2_3_B
           out_2_4_B = F.relu(self.bn1_2_4_B(self.conv1_2_4_B(x_2_4_B)))
           out_2_4_B = self.bn2_2_4_B(self.conv2_2_4_B(out_2_4_B))
           out_2_4_B += self.shortcut_2_4_B(x_2_4_B)
           out_2_4_B = F.relu(out_2_4_B)

           # out_2_4_B += out_2_4_A

           x_2_5_A = out_2_4_A
           out_2_5_A = F.relu(self.bn1_2_5_A(self.conv1_2_5_A(x_2_5_A)))
           out_2_5_A = self.bn2_2_5_A(self.conv2_2_5_A(out_2_5_A))
           out_2_5_A += self.shortcut_2_5_A(x_2_5_A)
           out_2_5_A = F.relu(out_2_5_A)

           x_2_5_B = out_2_4_B
           out_2_5_B = F.relu(self.bn1_2_5_B(self.conv1_2_5_B(x_2_5_B)))
           out_2_5_B = self.bn2_2_5_B(self.conv2_2_5_B(out_2_5_B))
           out_2_5_B += self.shortcut_2_5_B(x_2_5_B)
           out_2_5_B = F.relu(out_2_5_B)

           # out_2_5_B += out_2_5_A

           x_3_1_A = out_2_5_A
           out_3_1_A = F.relu(self.bn1_3_1_A(self.conv1_3_1_A(x_3_1_A)))
           out_3_1_A = self.bn2_3_1_A(self.conv2_3_1_A(out_3_1_A))
           out_3_1_A += self.shortcut_3_1_A(x_3_1_A)
           out_3_1_A = F.relu(out_3_1_A)

           x_3_1_B = out_2_5_B
           out_3_1_B = F.relu(self.bn1_3_1_B(self.conv1_3_1_B(x_3_1_B)))
           out_3_1_B = self.bn2_3_1_B(self.conv2_3_1_B(out_3_1_B))
           out_3_1_B += self.shortcut_3_1_B(x_3_1_B)
           out_3_1_B = F.relu(out_3_1_B)

           # out_3_1_B += out_3_1_A

           x_3_2_A = out_3_1_A
           out_3_2_A = F.relu(self.bn1_3_2_A(self.conv1_3_2_A(x_3_2_A)))
           out_3_2_A = self.bn2_3_2_A(self.conv2_3_2_A(out_3_2_A))
           out_3_2_A += self.shortcut_3_2_A(x_3_2_A)
           out_3_2_A = F.relu(out_3_2_A)

           x_3_2_B = out_3_1_B
           out_3_2_B = F.relu(self.bn1_3_2_B(self.conv1_3_2_B(x_3_2_B)))
           out_3_2_B = self.bn2_3_2_B(self.conv2_3_2_B(out_3_2_B))
           out_3_2_B += self.shortcut_3_2_B(x_3_2_B)
           out_3_2_B = F.relu(out_3_2_B)

           # out_3_2_B += out_3_2_A

           x_3_3_A = out_3_2_A
           out_3_3_A = F.relu(self.bn1_3_3_A(self.conv1_3_3_A(x_3_3_A)))
           out_3_3_A = self.bn2_3_3_A(self.conv2_3_3_A(out_3_3_A))
           out_3_3_A += self.shortcut_3_3_A(x_3_3_A)
           out_3_3_A = F.relu(out_3_3_A)

           x_3_3_B = out_3_2_B
           out_3_3_B = F.relu(self.bn1_3_3_B(self.conv1_3_3_B(x_3_3_B)))
           out_3_3_B = self.bn2_3_3_B(self.conv2_3_3_B(out_3_3_B))
           out_3_3_B += self.shortcut_3_3_B(x_3_3_B)
           out_3_3_B = F.relu(out_3_3_B)

           # out_3_3_B += out_3_3_A

           x_3_4_A = out_3_3_A
           out_3_4_A = F.relu(self.bn1_3_4_A(self.conv1_3_4_A(x_3_4_A)))
           out_3_4_A = self.bn2_3_4_A(self.conv2_3_4_A(out_3_4_A))
           out_3_4_A += self.shortcut_3_4_A(x_3_4_A)
           out_3_4_A = F.relu(out_3_4_A)

           x_3_4_B = out_3_3_B
           out_3_4_B = F.relu(self.bn1_3_4_B(self.conv1_3_4_B(x_3_4_B)))
           out_3_4_B = self.bn2_3_4_B(self.conv2_3_4_B(out_3_4_B))
           out_3_4_B += self.shortcut_3_4_B(x_3_4_B)
           out_3_4_B = F.relu(out_3_4_B)

           # out_3_4_B += out_3_4_A

           x_3_5_A = out_3_4_A
           out_3_5_A = F.relu(self.bn1_3_5_A(self.conv1_3_5_A(x_3_5_A)))
           out_3_5_A = self.bn2_3_5_A(self.conv2_3_5_A(out_3_5_A))
           out_3_5_A += self.shortcut_3_5_A(x_3_5_A)
           out_3_5_A = F.relu(out_3_5_A)

           x_3_5_B = out_3_4_B
           out_3_5_B = F.relu(self.bn1_3_5_B(self.conv1_3_5_B(x_3_5_B)))
           out_3_5_B = self.bn2_3_5_B(self.conv2_3_5_B(out_3_5_B))
           out_3_5_B += self.shortcut_3_5_B(x_3_5_B)
           out_3_5_B = F.relu(out_3_5_B)

           # out_3_5_B += out_3_5_A

           out_3_5_A = F.avg_pool2d(out_3_5_A, out_3_5_A.size()[3])
           out_3_5_A = out_3_5_A.view(out_3_5_A.size(0), -1)

           out_3_5_B = F.avg_pool2d(out_3_5_B, out_3_5_B.size()[3])
           out_3_5_B = out_3_5_B.view(out_3_5_B.size(0), -1)

           out_3_5_B = self.linear_A(torch.cat([out_3_5_A, out_3_5_B], dim = 1))
       return out_3_5_B


   def estimate_fisher(self, data_loader, sample_size, batch_size = args.batch_size):
        # sample loglikelihoods from the dataset.
        # data_loader = utils.get_data_loader(dataset, batch_size)
        loglikelihoods = []
        for x, y in data_loader:
            # x = x.view(batch_size, -1)
            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)

            loglikelihoods.append(F.log_softmax(self(x), dim=1)[range(batch_size), y.data])

            if len(loglikelihoods) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        loglikelihood_grads = zip(*[autograd.grad(l, self.parameters(), retain_graph=(i < len(loglikelihoods))
        ) for i, l in enumerate(loglikelihoods, 1)])

        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g ** 2).mean() for g in loglikelihood_grads]
        param_names = [
            n.replace('.', '__') for n, p in self.named_parameters()
        ]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}



   def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace('.', '__')
            self.register_buffer('{}_mean'.format(n), p.data.clone())
            self.register_buffer('{}_fisher'
                                 .format(n), fisher[n].data.clone())

   def ewc_loss(self, cuda):
        # try:
        losses = []
        for n, p in self.named_parameters():
            # print(n, p)
            # retrieve the consolidated mean and fisher information.
            n = n.replace('.', '__')
            mean = getattr(self, '{}_mean'.format(n))
            fisher = getattr(self, '{}_fisher'.format(n))
            # wrap mean and fisher in variables.
            mean = Variable(mean)
            fisher = Variable(fisher)
            # calculate a ewc loss. (assumes the parameter's prior as
            # gaussian distribution with the estimated mean and the
            # estimated cramer-rao lower bound variance, which is
            # equivalent to the inverse of fisher information)
            losses.append((fisher * (p-mean)**2).sum())

        ewc_loss = (args.lamda/2)*sum(losses)
        # print('ewc_loss', ewc_loss)
        return ewc_loss

        except AttributeError:
            print('No consolidate/EWC loss available')
            # # ewc loss is 0 if there's no consolidated parameters.
            return (
                Variable(torch.zeros(1)).cuda() if cuda else
                Variable(torch.zeros(1))
            )

   def _is_on_cuda(self):
        return next(self.parameters()).is_cuda




