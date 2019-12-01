#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 20180612 qing, adopted from melee.py
"""Define the arena map configs."""

from __future__ import absolute_import

from pysc2.maps import lib


class Arena(lib.Map):
  directory = "Arena"
  download = "https://github.com/Tencent-Game-AI/SC2Arena/releases/download/v0.1/sc2arena_maps.zip"
  players = 2
  game_steps_per_episode = 16 * 60 * 30  # 30 minute limit.


arena_maps = [
    "2StalkerA",
    "4MarineA",
    "4MarineB",
    "HydraMutalisk",
    "Immortal2Zealot",
    "ImmortalZealot",
]

for name in arena_maps:
  globals()[name] = type(name, (Arena,), dict(filename=name))
