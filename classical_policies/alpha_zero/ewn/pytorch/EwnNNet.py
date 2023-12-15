import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
from classical_policies.alpha_zero.utils import *
import sys
sys.path.append('..')


class EwnNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(EwnNNet, self).__init__()
        # each cube has a layer
        self.conv1 = nn.Conv2d(
            self.board_z,
            args.num_channels,
            3,
            stride=1,
            padding=1)
        self.conv2 = nn.Conv2d(
            args.num_channels,
            args.num_channels,
            3,
            stride=1,
            padding=1)
        self.conv3 = nn.Conv2d(
            args.num_channels,
            args.num_channels,
            3,
            stride=1)
        self.conv4 = nn.Conv2d(
            args.num_channels,
            args.num_channels,
            3,
            stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        # add the input of dice_roll(cube_num)
        self.fc1 = nn.Linear(args.num_channels *
                             (self.board_x - 4) * (self.board_y - 4) + game.ewn.cube_num, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s, dice_roll):
        # s: batch_size x board_x x board_y x board_z

        # s = s.view(-1, self.board_z, self.board_x, self.board_y)
        # batch_size x self.board_z x board_x x board_y
        s = s.permute(0, 3, 1, 2)
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))
        # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn3(self.conv3(s)))
        # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu(self.bn4(self.conv4(s)))
        # s = s.view(-1, self.args.num_channels *
        #            (self.board_x - 4) * (self.board_y - 4))
        s = s.reshape(-1, self.args.num_channels *
                      (self.board_x - 4) * (self.board_y - 4))

        # add the input of dice_roll(cube_num)
        s = torch.cat((s, dice_roll), dim=1)

        s = F.dropout(
            F.relu(
                self.fc_bn1(
                    self.fc1(s))),
            p=self.args.dropout,
            training=self.training)  # batch_size x 1024
        s = F.dropout(
            F.relu(
                self.fc_bn2(
                    self.fc2(s))),
            p=self.args.dropout,
            training=self.training)  # batch_size x 512

        # batch_size x action_size
        pi = self.fc3(s)
        # batch_size x 1
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
