import os
import cv2
import numpy as np
from collections import namedtuple, deque
from dataclasses import dataclass
from typing import Any
from random import seed
from threeviz.api import plot_3d, plot_pose, plot_line_seg
from PIL import Image, ImageSequence
import time
import random
import argh
import tensorflow as tf
import tensorflow.keras as kr


def anneal_prob(itr, maxitr, start_itr, start_prob):
    m = (1 - start_prob) / (maxitr - start_itr)
    b = start_prob
    return m * (itr - start_itr) + b


@dataclass
class SingleStep:
    st: Any
    stn: Any
    at: int
    rt: float
    done: bool


class Agent:
    def __init__(self, i=0, j=0):
        self.i = i
        self.j = j

    @property
    def loc(self):
        return (self.i, self.j)

    def xmove(self, direction):
        direction = 1 if direction > 0 else -1
        return Agent(self.i, self.j + direction)

    def ymove(self, direction):
        direction = 1 if direction > 0 else -1
        return Agent(self.i + direction, self.j)

    def __repr__(self):
        return str(self.loc)


class QLearning:
    def __init__(self, num_states, num_actions, lr=0.1, discount_factor=0.99):
        self.q = np.zeros(num_states, num_actions)
        self.a = lr
        self.g = discount_factor

    def update(self, st, at, rt, st1):
        q = self.q
        a = self.a
        g = self.g
        q[st, at] = (1 - a) * q[st, at] + a * (rt + g * np.max(q[st1]))


class Maze:
    def __init__(self, rows=4, columns=4):
        self.env = np.zeros(rows, columns)
        self.mousy = Agent(0, 0)

    def randomize_agent(self):
        X, Y = np.where(self.env == 0)
        i = random.randint(0, len(x) - 1)
        self.mousy.i = X[i]
        self.mousy.j = Y[i]

    def reset(self):
        self.mousy.i = 0
        self.mousy.j = 0

    def in_bounds(self, i, j):
        nr, nc = self.env.shape
        return i >= 0 and i < nr and j >= 0 and j < nc

    def agent_in_bounds(self, a):
        return self.in_bounds(a.i, a.j)

    def is_valid_new_agent(self, a):
        return self.agent_in_bounds(a)

    @property
    def all_actions(self):
        a = self.mousy
        return [
            a.xmove(1),
            a.xmove(-1),
            a.ymove(1),
            a.ymove(-1),
        ]

    def apply_action(self, idx):
        moves = self.all_actions
        assert idx >= 0 and idx < len(
            moves), f"Index {idx} is not a valid action"
        move = moves[idx]
        score = -0.01
        win_score = 1
        death_score = -1
        if not self.is_valid_new_agent(move):
            return score, False
        self.do_a_move(move)
        if self.has_won():
            return win_score, True
        if self.has_died(move):
            return death_score, True

        return score, False

    def do_a_move(self, a):
        assert self.is_valid_new_agent(a), "Mousy can't go there"
        self.mousy = a
        return 10 if self.has_won() else -0.1

    def has_won(self):
        a = self.mousy
        return self.env[a.i, a.j] == 1

    def has_died(self):
        a = self.mousy
        return self.env[a.i, a.j] == -1

    def has_ended(self):
        return self.has_won() or self.has_died()

    def visualize(self):
        nr, nc = self.env.shape
        z = -0.1
        a = self.mousy
        plot_line_seg(0, 0, z, nr, 0, z, 'e1', size=0.2, color='red')
        plot_line_seg(0, 0, z, 0, nc, z, 'e2', size=0.2, color='red')
        plot_line_seg(0, nc, z, nr, nc, z, 'e3', size=0.2, color='red')
        plot_line_seg(nr, 0, z, nr, nc, z, 'e4', size=0.2, color='red')
        plot_3d(*get_midpoint_for_loc(a.i, a.j),
                z, 'mousy', color='blue', size=1)
        plot_3d(*get_midpoint_for_loc(nr - 1, nc - 1),
                z, 'goal', color='green', size=1)

        xarr, yarr = np.where(self.env == -1)
        plot_3d(xarr + 0.5, yarr + 0.5, [z] * len(xarr), 'obstacles', size=1.0)

    def to_image(self, image_shape=64):
        a = self.mousy
        e = self.env
        imout = np.expand_dims(np.ones_like(e) * 255, -1).astype('uint8')
        imout = np.dstack(imout, imout, imout)
        imout[e == -1, :] = 0
        imout[a.i, a.j, : - 1] = 0
        imout[-1, -1, ::2] = 0
        return cv2.resize(imout, (image_shape, image_shape), interpolation=cv2.INTER_NEAREST)

    def get_midpoint_for_loc(i, j):
        return i + 0.5, j + 0.5

    def make_test_maze(s=4):
        m = Maze(s, s)
        e = m.env
        h, w = e.shape
        e[-1, -1] = 1
        for i in range(len(e)):
            for j in range(len(e[i])):
                if i in [0, h - 1] and j in [0, w - 1]:
                    continue
                if random.random() < 0.3:
                    e[i, j] = -1

        return m
