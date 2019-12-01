from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNet(nn.Module):

  def __init__(self,
               resolution,
               n_channels,
               n_dims,
               n_out,
               batchnorm=False):
    super(DuelingQNet, self).__init__()
    assert resolution == 16
    self.conv1 = nn.Conv2d(in_channels=n_channels,
                           out_channels=32,
                           kernel_size=5,
                           stride=1,
                           padding=2)
    self.conv2 = nn.Conv2d(in_channels=32,
                           out_channels=32,
                           kernel_size=3,
                           stride=1,
                           padding=1)
    self.conv3 = nn.Conv2d(in_channels=32,
                           out_channels=16,
                           kernel_size=3,
                           stride=2,
                           padding=1)
    if batchnorm:
      self.bn1 = nn.BatchNorm2d(32)
      self.bn2 = nn.BatchNorm2d(32)
      self.bn3 = nn.BatchNorm2d(16)

    self.value_sp_fc = nn.Linear(16 * 8 * 8, 256)
    self.value_nonsp_fc1 = nn.Linear(n_dims, 512)
    self.value_nonsp_fc2 = nn.Linear(512, 512)
    self.value_nonsp_fc3 = nn.Linear(512, 256)
    self.value_final_fc = nn.Linear(512, 1)

    self.adv_sp_fc = nn.Linear(16 * 8 * 8, 256)
    self.adv_nonsp_fc1 = nn.Linear(n_dims, 512)
    self.adv_nonsp_fc2 = nn.Linear(512, 512)
    self.adv_nonsp_fc3 = nn.Linear(512, 256)
    self.adv_final_fc = nn.Linear(512, n_out)
    self._batchnorm = batchnorm

  def forward(self, x):
    spatial, nonspatial = x
    if self._batchnorm:
      spatial = F.relu(self.bn1(self.conv1(spatial)))
      spatial = F.relu(self.bn2(self.conv2(spatial)))
      spatial = F.relu(self.bn3(self.conv3(spatial)))
    else:
      spatial = F.relu(self.conv1(spatial))
      spatial = F.relu(self.conv2(spatial))
      spatial = F.relu(self.conv3(spatial))
    spatial = spatial.view(spatial.size(0), -1)

    value_sp_state = F.relu(self.value_sp_fc(spatial))
    value_nonsp_state = F.relu(self.value_nonsp_fc1(nonspatial))
    value_nonsp_state = F.relu(self.value_nonsp_fc2(value_nonsp_state))
    value_nonsp_state = F.relu(self.value_nonsp_fc3(value_nonsp_state))
    value_state = torch.cat((value_sp_state, value_nonsp_state), 1)
    value = self.value_final_fc(value_state)

    adv_sp_state = F.relu(self.adv_sp_fc(spatial))
    adv_nonsp_state = F.relu(self.adv_nonsp_fc1(nonspatial))
    adv_nonsp_state = F.relu(self.adv_nonsp_fc2(adv_nonsp_state))
    adv_nonsp_state = F.relu(self.adv_nonsp_fc3(adv_nonsp_state))
    adv_state = torch.cat((adv_sp_state, adv_nonsp_state), 1)
    adv = self.adv_final_fc(adv_state)
    adv_subtract = adv - adv.mean(dim=1, keepdim=True)
    return value + adv_subtract


class NonspatialDuelingQNet(nn.Module):

  def __init__(self, n_dims, n_out):
    super(NonspatialDuelingQNet, self).__init__()
    self.value_fc1 = nn.Linear(n_dims, 512)
    self.value_fc2 = nn.Linear(512, 512)
    self.value_fc3 = nn.Linear(512, 256)
    self.value_fc4 = nn.Linear(256, 1)

    self.adv_fc1 = nn.Linear(n_dims, 512)
    self.adv_fc2 = nn.Linear(512, 512)
    self.adv_fc3 = nn.Linear(512, 256)
    self.adv_fc4 = nn.Linear(256, n_out)

  def forward(self, x):
    value = F.relu(self.value_fc1(x))
    value = F.relu(self.value_fc2(value))
    value = F.relu(self.value_fc3(value))
    value = self.value_fc4(value)

    adv = F.relu(self.adv_fc1(x))
    adv = F.relu(self.adv_fc2(adv))
    adv = F.relu(self.adv_fc3(adv))
    adv = self.adv_fc4(adv)

    adv_subtract = adv - adv.mean(dim=1, keepdim=True)
    return value + adv_subtract
