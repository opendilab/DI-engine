# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Expose static data in a more useful form than the raw protos."""

import six
import torch
import numpy as np
from .action_dict import ACTION_INFO_MASK, GENERAL_ACTION_INFO_MASK, ACTIONS_STAT


class StaticData(object):
    """Expose static data in a more useful form than the raw protos."""

    def __init__(self, data):
        """Takes data from RequestData."""
        self._units = {u.unit_id: u.name for u in data.units}
        self._unit_stats = {u.unit_id: u for u in data.units}
        self._upgrades = {a.upgrade_id: a for a in data.upgrades}
        self._abilities = {a.ability_id: a for a in data.abilities}
        self._general_abilities = {a.remaps_to_ability_id for a in data.abilities if a.remaps_to_ability_id}

        for a in six.itervalues(self._abilities):
            a.hotkey = a.hotkey.lower()

    @property
    def abilities(self):
        return self._abilities

    @property
    def upgrades(self):
        return self._upgrades

    @property
    def units(self):
        return self._units

    @property
    def unit_stats(self):
        return self._unit_stats

    @property
    def general_abilities(self):
        return self._general_abilities


def get_reorder_lookup_array(input):
    num = max(input)
    # use -1 as marker for invalid entry
    array = torch.full((num + 1, ), -1, dtype=torch.long)
    for index, item in enumerate(input):
        array[item] = index
    return array


# List of used/available abilities found by parsing replays.
ABILITIES = [
    0,  # invalid
    1,
    4,
    6,
    7,
    16,
    17,
    18,
    19,
    23,
    26,
    28,
    30,
    32,
    36,
    38,
    42,
    44,
    46,
    74,
    76,
    78,
    80,
    110,
    140,
    142,
    144,
    146,
    148,
    150,
    152,
    154,
    156,
    158,
    160,
    162,
    164,
    166,
    167,
    169,
    171,
    173,
    174,
    181,
    195,
    199,
    203,
    207,
    211,
    212,
    216,
    217,
    247,
    249,
    250,
    251,
    253,
    255,
    261,
    263,
    265,
    295,
    296,
    298,
    299,
    304,
    305,
    306,
    307,
    308,
    309,
    312,
    313,
    314,
    315,
    316,
    318,
    319,
    320,
    321,
    322,
    323,
    324,
    326,
    327,
    328,
    329,
    331,
    333,
    348,
    380,
    382,
    383,
    386,
    388,
    390,
    392,
    393,
    394,
    396,
    397,
    399,
    401,
    403,
    405,
    407,
    408,
    410,
    413,
    415,
    416,
    417,
    419,
    421,
    422,
    451,
    452,
    454,
    455,
    484,
    485,
    487,
    488,
    517,
    518,
    520,
    522,
    524,
    554,
    556,
    558,
    560,
    561,
    562,
    563,
    591,
    594,
    595,
    596,
    597,
    614,
    620,
    621,
    622,
    623,
    624,
    626,
    650,
    651,
    652,
    653,
    654,
    655,
    656,
    657,
    658,
    710,
    730,
    731,
    732,
    761,
    764,
    766,
    768,
    769,
    790,
    793,
    799,
    803,
    804,
    805,
    820,
    822,
    855,
    856,
    857,
    861,
    862,
    863,
    864,
    865,
    866,
    880,
    881,
    882,
    883,
    884,
    885,
    886,
    887,
    889,
    890,
    891,
    892,
    893,
    894,
    895,
    911,
    913,
    914,
    916,
    917,
    919,
    920,
    921,
    922,
    946,
    948,
    950,
    954,
    955,
    976,
    977,
    978,
    979,
    994,
    1006,
    1036,
    1038,
    1039,
    1042,
    1062,
    1063,
    1064,
    1065,
    1066,
    1067,
    1068,
    1069,
    1070,
    1093,
    1094,
    1097,
    1126,
    1152,
    1154,
    1155,
    1156,
    1157,
    1158,
    1159,
    1160,
    1161,
    1162,
    1163,
    1165,
    1166,
    1167,
    1183,
    1184,
    1186,
    1187,
    1188,
    1189,
    1190,
    1191,
    1192,
    1193,
    1194,
    1216,
    1217,
    1218,
    1219,
    1220,
    1221,
    1223,
    1225,
    1252,
    1253,
    1282,
    1283,
    1312,
    1313,
    1314,
    1315,
    1316,
    1317,
    1342,
    1343,
    1344,
    1345,
    1346,
    1348,
    1351,
    1352,
    1353,
    1354,
    1356,
    1372,
    1373,
    1374,
    1376,
    1378,
    1380,
    1382,
    1384,
    1386,
    1388,
    1390,
    1392,
    1394,
    1396,
    1406,
    1408,
    1409,
    1413,
    1414,
    1416,
    1417,
    1418,
    1419,
    1433,
    1435,
    1437,
    1438,
    1440,
    1442,
    1444,
    1446,
    1448,
    1449,
    1450,
    1451,
    1454,
    1455,
    1482,
    1512,
    1514,
    1516,
    1517,
    1518,
    1520,
    1522,
    1524,
    1526,
    1528,
    1530,
    1532,
    1562,
    1563,
    1564,
    1565,
    1566,
    1567,
    1568,
    1592,
    1593,
    1594,
    1622,
    1623,
    1628,
    1632,
    1664,
    1682,
    1683,
    1684,
    1691,
    1692,
    1693,
    1694,
    1725,
    1727,
    1729,
    1730,
    1731,
    1732,
    1733,
    1763,
    1764,
    1766,
    1768,
    1819,
    1825,
    1831,
    1832,
    1833,
    1834,
    1847,
    1848,
    1853,
    1974,
    1978,
    1998,
    2014,
    2016,
    2048,
    2057,
    2063,
    2067,
    2073,
    2081,
    2082,
    2095,
    2097,
    2099,
    2108,
    2110,
    2112,
    2113,
    2114,
    2116,
    2146,
    2162,
    2244,
    2324,
    2328,
    2330,
    2331,
    2332,
    2333,
    2338,
    2340,
    2342,
    2346,
    2350,
    2354,
    2358,
    2362,
    2364,
    2365,
    2368,
    2370,
    2371,
    2373,
    2375,
    2376,
    2387,
    2389,
    2391,
    2393,
    2505,
    2535,
    2542,
    2544,
    2550,
    2552,
    2558,
    2560,
    2588,
    2594,
    2596,
    2700,
    2704,
    2708,
    2709,
    2714,
    2720,
    3707,
    3709,
    3739,
    3741,
    3743,
    3745,
    3747,
    3749,
    3751,
    3753,
    3755,
    3757,
    3765,
    3771,
    3776,
    3777,
    3778,
    3783,
]

#    356, 503, 547, 360, 515, 193, 10, 197, 528, 495, 516, 184, 491, 190, 483,  # TODO
#    498, 192, 215, 189, 437, 519, 514, 219, 198, 507, 204, 400, 349, 492, 431,
#    543, 201, 387, 442, 479, 551, 489, 425, 218, 447, 238, 220, 501, 391, 445,
#    438, 526, 350, 256, 494, 493,

NUM_ABILITIES = len(ABILITIES)

ABILITIES_REORDER = {item: idx for idx, item in enumerate(ABILITIES)}

ABILITIES_REORDER_ARRAY = get_reorder_lookup_array(ABILITIES)

# List of known unit types. It is generated by parsing replays and from:
# https://github.com/Blizzard/s2client-api/blob/master/include/sc2api/sc2_typeenums.h
UNIT_TYPES = [
    0,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
    117,
    118,
    119,
    120,
    125,
    126,
    127,
    128,
    129,
    130,
    131,
    132,
    133,
    134,
    135,
    136,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    149,
    150,
    151,
    268,
    289,
    311,
    321,
    322,
    324,
    330,
    335,
    336,
    341,
    342,
    343,
    344,
    350,
    364,
    365,
    371,
    372,
    373,
    376,
    377,
    472,
    473,
    474,
    475,
    483,
    484,
    485,
    486,
    487,
    488,
    489,
    490,
    493,
    494,
    495,
    496,
    498,
    499,
    500,
    501,
    502,
    503,
    504,
    517,
    518,
    559,
    560,
    561,
    562,
    563,
    564,
    588,
    589,
    590,
    591,
    608,
    609,
    610,
    612,
    628,
    629,
    630,
    638,
    639,
    640,
    641,
    642,
    643,
    648,
    649,
    651,
    661,
    662,
    663,
    664,
    665,
    666,
    687,
    688,
    689,
    690,
    691,
    692,
    693,
    694,
    732,
    733,
    734,
    796,
    797,
    801,
    824,
    830,
    877,
    880,
    881,
    884,
    885,
    886,
    887,
    892,
    893,
    894,
    1904,
    1908,
    1910,
    1911,
    1912,
    1913,
    1955,
    1956,
    1957,
    1958,
    1960,
    1961,
]

UNIT_TYPES_REORDER = {item: idx for idx, item in enumerate(UNIT_TYPES)}

NUM_UNIT_TYPES = len(UNIT_TYPES)

UNIT_TYPES_REORDER_ARRAY = get_reorder_lookup_array(UNIT_TYPES)

# List of used buffs found by parsing replays.
BUFFS = [
    0,  # TODO
    5,
    6,
    7,
    8,
    11,
    12,
    13,
    16,
    17,
    18,
    22,
    24,
    25,
    27,
    28,
    29,
    30,
    33,
    36,
    38,
    49,
    59,
    83,
    89,
    99,
    102,
    116,
    121,
    122,
    129,
    132,
    133,
    134,
    136,
    137,
    145,
    271,
    272,
    273,
    274,
    275,
    277,
    279,
    280,
    281,
    288,
    289,
    20,
    97,
]

NUM_BUFFS = len(BUFFS)

BUFFS_REORDER = {item: idx for idx, item in enumerate(BUFFS)}

BUFFS_REORDER_ARRAY = get_reorder_lookup_array(BUFFS)

# List of used upgrades found by parsing replays.
UPGRADES = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    16,
    17,
    19,
    20,
    22,
    25,
    30,
    31,
    32,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    64,
    65,
    66,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    86,
    87,
    88,
    99,
    101,
    116,
    117,
    118,
    122,
    130,
    134,
    135,
    136,
    139,
    140,
    141,
    144,
    289,
    291,
    293,
    296,
]

NUM_UPGRADES = len(UPGRADES)

UPGRADES_REORDER = {item: idx for idx, item in enumerate(UPGRADES)}

UPGRADES_REORDER_ARRAY = get_reorder_lookup_array(UPGRADES)

UPGRADES_REORDER_INV = {v: k for k, v in UPGRADES_REORDER.items()}

UPGRADES_REORDER_INV_ARRAY = UPGRADES

ADDON = [0, 5, 6, 37, 38, 39, 40, 41, 42]

NUM_ADDON = len(ADDON)

ADDON_REORDER = {item: idx for idx, item in enumerate(ADDON)}

ADDON_REORDER_ARRAY = get_reorder_lookup_array(ADDON)

ACTIONS = list(GENERAL_ACTION_INFO_MASK.keys())

NUM_ACTIONS = len(ACTIONS)

NUM_ACTIONS_RAW = len(list(ACTION_INFO_MASK.keys()))

# this is for the translation from distar.ctools.pysc2 ability id to distar.ctools.pysc2 raw ability id
# _FUNCTIONS:
# https://github.com/deepmind/ctools.pysc2/blob/5ca04dbf6dd0b852966418379e2d95d9ad3393f8/ctools.pysc2/lib/actions.py#L583
# _RAW_FUNCTIONS:
# https://github.com/deepmind/ctools.pysc2/blob/5ca04dbf6dd0b852966418379e2d95d9ad3393f8/ctools.pysc2/lib/actions.py#L1186
# ACTIONS_REORDER[ctools.pysc2_ability_id] = distar.ctools.pysc2_raw_ability_id
ACTIONS_REORDER = {item: idx for idx, item in enumerate(ACTIONS)}

ACTIONS_REORDER_ARRAY = get_reorder_lookup_array(ACTIONS)

ACTIONS_REORDER_INV = {v: k for k, v in ACTIONS_REORDER.items()}

ACTIONS_REORDER_INV_ARRAY = ACTIONS

# Begin actions: actions (raw ability ids) included in the beginning_build_order
target_list = ['unit', 'build', 'research']
BEGIN_ACTIONS = [k for k, v in GENERAL_ACTION_INFO_MASK.items() if v['goal'] in target_list]
OLD_BEGIN_ACTIONS = [k for k, v in GENERAL_ACTION_INFO_MASK.items() if v['goal'] in target_list + ['effect']]

removed_actions = [35, 64, 520, 222, 515, 503]
for i in removed_actions:
    BEGIN_ACTIONS.remove(i)
NUM_BEGIN_ACTIONS = len(BEGIN_ACTIONS)

OLD_BEGIN_ACTIONS_REORDER = {item: idx for idx, item in enumerate(OLD_BEGIN_ACTIONS)}

OLD_BEGIN_ACTIONS_REORDER_INV = {v: k for k, v in OLD_BEGIN_ACTIONS_REORDER.items()}

BUILD_ORDER_REWARD_ACTIONS = BEGIN_ACTIONS

BEGIN_ACTIONS_REORDER_ARRAY = get_reorder_lookup_array(BEGIN_ACTIONS)

UNIT_BUILD_ACTIONS = [k for k, v in GENERAL_ACTION_INFO_MASK.items() if v['goal'] in ['unit', 'build']]

NUM_UNIT_BUILD_ACTIONS = len(UNIT_BUILD_ACTIONS)

UNIT_BUILD_ACTIONS_REORDER = {item: idx for idx, item in enumerate(UNIT_BUILD_ACTIONS)}

UNIT_BUILD_ACTIONS_REORDER_ARRAY = get_reorder_lookup_array(UNIT_BUILD_ACTIONS)

EFFECT_ACTIONS = [k for k, v in GENERAL_ACTION_INFO_MASK.items() if v['goal'] in ['effect']]

NUM_EFFECT_ACTIONS = len(EFFECT_ACTIONS)

EFFECT_ACTIONS_REORDER = {item: idx for idx, item in enumerate(EFFECT_ACTIONS)}

EFFECT_ACTIONS_REORDER_ARRAY = get_reorder_lookup_array(EFFECT_ACTIONS)

RESEARCH_ACTIONS = [k for k, v in GENERAL_ACTION_INFO_MASK.items() if v['goal'] in ['research']]

NUM_RESEARCH_ACTIONS = len(RESEARCH_ACTIONS)

RESEARCH_ACTIONS_REORDER = {item: idx for idx, item in enumerate(RESEARCH_ACTIONS)}

RESEARCH_ACTIONS_REORDER_ARRAY = get_reorder_lookup_array(RESEARCH_ACTIONS)

ORDER_ACTIONS = [
    425, 85, 426, 553, 427, 428, 429, 84, 430, 431, 83, 432, 433, 434, 554, 435, 436, 563, 69, 437, 67, 68, 438, 440,
    439, 441, 18, 442, 443, 444, 445, 446, 19, 447, 116, 117, 118, 119, 120, 70, 448, 449, 97, 450, 451, 452, 456, 460,
    464, 465, 469, 473, 82, 474, 478, 482, 494, 495, 486, 490, 54, 499, 500, 56, 62, 501, 502, 52, 166, 504, 505, 506,
    51, 63, 509, 510, 511, 512, 513, 21, 61, 58, 55, 64, 516, 517, 518, 520, 53, 521, 50, 59, 523, 525, 57, 76, 74, 73,
    60, 75, 72, 71, 527, 49
]

NUM_ORDER_ACTIONS = len(ORDER_ACTIONS) + 1
ORDER_ACTIONS_REORDER_ARRAY = torch.zeros(max(GENERAL_ACTION_INFO_MASK.keys()) + 1, dtype=torch.long)
for idx, v in enumerate(ORDER_ACTIONS):
    ORDER_ACTIONS_REORDER_ARRAY[v] = idx + 1


def ger_reorder_tag(val, template):
    low = 0
    high = len(template)
    while low < high:
        mid = (low + high) // 2
        mid_val = template[mid]
        if val == mid_val:
            return mid
        elif val > mid_val:
            low = mid + 1
        else:
            high = mid
    raise ValueError("unknow found val: {}".format(val))


SELECTED_UNITS_MASK = torch.zeros(NUM_ACTIONS, NUM_UNIT_TYPES)
TARGET_UNITS_MASK = torch.zeros(NUM_ACTIONS, NUM_UNIT_TYPES)
for i in range(NUM_ACTIONS):
    a = ACTIONS_REORDER_INV[i]
    type_set = set(ACTIONS_STAT[a]['selected_type'])
    reorder_type_list = [UNIT_TYPES_REORDER[i] for i in type_set]
    SELECTED_UNITS_MASK[i, reorder_type_list] = 1
    type_set = set(ACTIONS_STAT[a]['target_type'])
    reorder_type_list = [UNIT_TYPES_REORDER[i] for i in type_set]
    TARGET_UNITS_MASK[i, reorder_type_list] = 1

UNIT_SPECIFIC_ABILITIES = [
    4, 6, 7, 9, 2063, 16, 17, 18, 19, 2067, 23, 2073, 26, 28, 30, 2081, 36, 38, 40, 42, 46, 2095, 2097, 2099, 2108,
    2110, 2116, 74, 76, 78, 80, 2146, 110, 140, 142, 146, 148, 152, 154, 166, 167, 173, 181, 2244, 216, 217, 247, 249,
    251, 253, 263, 265, 2324, 2330, 2332, 2338, 2340, 2342, 295, 296, 2346, 298, 299, 2350, 2358, 2362, 316, 2364, 318,
    319, 320, 321, 322, 323, 324, 2370, 326, 2375, 2376, 327, 328, 329, 331, 333, 2383, 2385, 2387, 2393, 380, 382, 383,
    386, 388, 390, 392, 393, 394, 396, 397, 401, 403, 405, 407, 412, 413, 416, 417, 419, 421, 422, 452, 454, 455, 2505,
    485, 487, 488, 2536, 2542, 2544, 2554, 2556, 2558, 2560, 518, 520, 522, 524, 2588, 2594, 2596, 554, 556, 558, 560,
    561, 562, 563, 591, 594, 595, 596, 597, 614, 620, 621, 622, 623, 624, 626, 650, 651, 652, 653, 654, 2700, 656, 657,
    2706, 658, 2708, 2704, 2714, 2720, 710, 730, 731, 732, 761, 764, 766, 769, 790, 793, 799, 804, 805, 820, 855, 856,
    857, 861, 862, 863, 864, 865, 866, 880, 881, 882, 883, 884, 885, 886, 887, 889, 890, 891, 892, 893, 894, 895, 911,
    913, 914, 916, 917, 919, 920, 921, 922, 946, 948, 950, 954, 955, 976, 977, 978, 979, 994, 1006, 1036, 1042, 1062,
    1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1093, 1094, 1097, 1126, 1152, 1154, 1155, 1156, 1157, 1158, 1159,
    1160, 1161, 1162, 1163, 1165, 1166, 1167, 1183, 1184, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1216,
    1218, 1220, 1223, 1225, 1252, 1253, 1282, 1283, 1312, 1313, 1314, 1315, 1316, 1317, 1342, 1343, 1344, 1345, 1346,
    1348, 1351, 1352, 1353, 1354, 1356, 1372, 1374, 1376, 1378, 1380, 1382, 1384, 1386, 1388, 1390, 1392, 1394, 1396,
    1406, 1408, 1409, 1433, 1435, 1437, 1438, 1440, 1442, 1444, 1446, 1448, 1450, 1454, 1455, 1482, 1512, 1514, 1516,
    1518, 1520, 1522, 1524, 1526, 1528, 1530, 1532, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1592, 1593, 1594, 1622,
    1628, 1632, 3707, 3709, 1662, 1664, 3739, 1692, 1693, 1694, 3741, 3743, 3745, 3747, 3753, 3765, 3771, 1725, 1727,
    3776, 1729, 3778, 1731, 3777, 1733, 3783, 1764, 1767, 1768, 1819, 1825, 1978, 1998, 2014, 2016
]

# correspond to UNIT_SPECIFIC_ABILITIES
UNIT_GENERAL_ABILITIES = [
    3665, 3665, 3665, 3665, 0, 3794, 3795, 3793, 3674, 0, 3674, 0, 3684, 3684, 3684, 0, 3688, 3689, 0, 0, 0, 3661, 3662,
    0, 3661, 3662, 0, 0, 0, 3685, 0, 0, 0, 0, 3686, 0, 0, 0, 0, 3666, 3667, 0, 0, 0, 0, 0, 0, 0, 0, 3675, 0, 0, 0, 0, 0,
    0, 3661, 3662, 3666, 3667, 0, 3666, 3667, 0, 0, 0, 3685, 0, 0, 0, 0, 0, 0, 0, 0, 3668, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 3675, 3676, 3677, 0, 0, 0, 3676, 3677, 3668, 3669, 3796, 0, 0, 0, 3668, 3668, 3664, 3663, 3679, 3678, 3682,
    3683, 3679, 3682, 3683, 0, 3679, 3682, 3683, 0, 0, 0, 0, 0, 0, 0, 3679, 3678, 3678, 0, 0, 3659, 3659, 3678, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3698, 3698, 3698, 3687, 3697, 3697, 0, 3697, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3701, 3701, 3701, 3699, 3699, 3699, 3700, 3700, 3700, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 3668, 3669, 3796, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3695, 3695, 3695, 3694,
    3694, 3694, 3696, 3696, 3696, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3666, 3667, 3705, 3705, 3705,
    3704, 3704, 3704, 3706, 3706, 3706, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3703, 3703, 3703, 3702, 3702, 3702, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 3661, 3662, 3661, 3662, 3661, 3662, 3661, 3662, 3661, 3662, 3661, 3662, 3668, 3669, 3796, 3661,
    3662, 3668, 3664, 3796, 3687, 3661, 3662, 0, 0, 0, 0, 0, 3661, 3662, 0, 0, 0, 3679, 3678, 0, 0, 0, 0, 3693, 3693,
    3693, 3692, 3692, 3692, 0, 0, 0, 0, 0, 0, 0, 3659, 0, 3661, 0, 0, 0, 0, 3691, 0, 0, 0, 0, 0, 0, 3674, 3681, 3681,
    3794, 3680, 3793, 3680, 3795, 3691, 3665, 0, 0, 0, 0, 0, 0, 0, 3661, 3662
]

# UNIT_MIX_ABILITIES = []
# for idx in range(len(UNIT_SPECIFIC_ABILITIES)):
#     if UNIT_GENERAL_ABILITIES[idx] == 0:
#         UNIT_MIX_ABILITIES.append(UNIT_SPECIFIC_ABILITIES[idx])
#     else:
#         UNIT_MIX_ABILITIES.append(UNIT_GENERAL_ABILITIES[idx])
# UNIT_MIX_ABILITIES = [0] + list(set(UNIT_MIX_ABILITIES))  # use 0 as no op

UNIT_MIX_ABILITIES = [
    0, 2560, 524, 1036, 2063, 1042, 2067, 1530, 2073, 1344, 2588, 1532, 1568, 2081, 1126, 40, 42, 556, 46, 558, 560,
    561, 562, 2099, 563, 1345, 1592, 1593, 1594, 2116, 1093, 1094, 1097, 74, 3659, 76, 3661, 3662, 3663, 3664, 80, 3665,
    3666, 3667, 3669, 3668, 591, 594, 595, 3674, 3675, 3676, 3677, 3678, 3679, 1628, 1632, 2146, 3682, 3684, 3685, 3686,
    3683, 3688, 3689, 614, 3687, 620, 621, 110, 622, 623, 624, 626, 3698, 3697, 3701, 3699, 3700, 3695, 3696, 3705,
    3704, 3706, 3703, 3702, 1348, 1152, 3709, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 650, 651, 140, 1162, 1163,
    1165, 2704, 1166, 146, 2706, 148, 2708, 1167, 3693, 152, 3692, 154, 2714, 3694, 1354, 3739, 1692, 2720, 1693, 3741,
    3743, 3745, 3747, 3753, 173, 181, 1664, 3765, 3691, 1216, 1218, 2244, 1220, 710, 1223, 1225, 3793, 3794, 3795, 3796,
    216, 217, 730, 731, 732, 1252, 1253, 1764, 1767, 1768, 247, 249, 761, 251, 764, 766, 769, 1282, 1283, 263, 265,
    2324, 790, 793, 2330, 1819, 2332, 799, 1825, 2338, 804, 805, 2346, 2350, 820, 2358, 2362, 2364, 318, 319, 320, 321,
    322, 323, 324, 1342, 326, 2375, 2376, 327, 328, 331, 329, 333, 1351, 2383, 1352, 2385, 1353, 2387, 1356, 2393, 1372,
    880, 881, 882, 883, 884, 885, 886, 887, 1346, 889, 890, 891, 892, 893, 894, 895, 386, 388, 390, 401, 403, 916, 405,
    917, 919, 920, 921, 922, 1448, 3680, 1450, 1454, 1455, 946, 948, 950, 596, 954, 955, 597, 1978, 3681, 2505, 1482,
    1998, 976, 977, 978, 979, 1622, 994, 2536, 1516, 2542, 1006, 2544, 1518, 1520, 1526, 1528, 2554, 1343, 2556, 2558
]
