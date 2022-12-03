

GAME_CONSTANTS = {
  "UNIT_TYPES": {
    "WORKER": 0,
    "CART": 1
  },
  "RESOURCE_TYPES": {
    "WOOD": "wood",
    "COAL": "coal",
    "URANIUM": "uranium"
  },
  "DIRECTIONS": {
    "NORTH": "n",
    "WEST": "w",
    "EAST": "e",
    "SOUTH": "s",
    "CENTER": "c"
  },
  "PARAMETERS": {
    "DAY_LENGTH": 30,
    "NIGHT_LENGTH": 10,
    "MAX_DAYS": 360,
    "LIGHT_UPKEEP": {
      "CITY": 23,
      "WORKER": 4,
      "CART": 10
    },
    "WOOD_GROWTH_RATE": 1.025,
    "MAX_WOOD_AMOUNT": 500,
    "CITY_BUILD_COST": 100,
    "CITY_ADJACENCY_BONUS": 5,
    "RESOURCE_CAPACITY": {
      "WORKER": 100,
      "CART": 2000
    },
    "WORKER_COLLECTION_RATE": {
      "WOOD": 20,
      "COAL": 5,
      "URANIUM": 2
    },
    "RESOURCE_TO_FUEL_RATE": {
      "WOOD": 1,
      "COAL": 10,
      "URANIUM": 40
    },
    "RESEARCH_REQUIREMENTS": {
      "COAL": 50,
      "URANIUM": 200
    },
    "CITY_ACTION_COOLDOWN": 10,
    "UNIT_ACTION_COOLDOWN": {
      "CART": 3,
      "WORKER": 2
    },
    "MAX_ROAD": 6,
    "MIN_ROAD": 0,
    "CART_ROAD_DEVELOPMENT_RATE": 0.75,
    "PILLAGE_RATE": 0.5
  }
}

class Constants:
    class AGENT_TYPE:
        AGENT = "agent"
        LEARNING = "learning"

    class INPUT_CONSTANTS:
        RESEARCH_POINTS = "rp"
        RESOURCES = "r"
        UNITS = "u"
        CITY = "c"
        CITY_TILES = "ct"
        ROADS = "ccd"
        DONE = "D_DONE"

    class DIRECTIONS:
        NORTH = "n"
        WEST = "w"
        SOUTH = "s"
        EAST = "e"
        CENTER = "c"

    class UNIT_TYPES:
        WORKER = 0
        CART = 1

    class TEAM:
        A = 0
        B = 1

    class RESOURCE_TYPES:
        WOOD = "wood"
        URANIUM = "uranium"
        COAL = "coal"

    class MAP_TYPES:
        EMPTY = 'empty'
        RANDOM = 'random'
        DEBUG = 'debug'

    # Mirrored Game constant enums. All the available agent actions with specifications as to what they do and restrictions.
    class ACTIONS:
        #
        # Formatted as `m unit_id direction`. unit_id should be valid and should have empty space in that direction. moves
        # unit with id unit_id in the direction
        #
        MOVE = 'm'
        #
        # Formatted as `r x y`. (x,y) should be an owned city tile, the city tile is commanded to research for
        # the next X turns
        # /
        RESEARCH = 'r'
        # Formatted as `bw x y`. (x,y) should be an owned city tile, where worker is to be built #/
        BUILD_WORKER = 'bw'
        # Formatted as `bc x y`. (x,y) should be an owned city tile, where the cart is to be built #/
        BUILD_CART = 'bc'
        #
        # Formatted as `bcity unit_id`. builds city at unit_id's pos, unit_id should be
        # friendly owned unit that is a worker
        # /
        BUILD_CITY = 'bcity'
        #
        # Formatted as `t source_unit_id destination_unit_id resource_type amount`. Both units in transfer should be
        # adjacent. If command valid, it will transfer as much as possible with a max of the amount specified
        # /
        TRANSFER = 't'

        # formatted as `p unit_id`. Unit with the given unit_id must be owned and pillages the tile they are on #/
        PILLAGE = 'p'

        # formatted as dc <x> <y> #/
        DEBUG_ANNOTATE_CIRCLE = 'dc'
        # formatted as dx <x> <y> #/
        DEBUG_ANNOTATE_X = 'dx'
        # formatted as dl <x1> <y1> <x2> <y2> #/
        DEBUG_ANNOTATE_LINE = 'dl'
        # formatted as dt <x> <y> <message> <fontsize> #/
        DEBUG_ANNOTATE_TEXT = 'dt'
        # formatted as dst <message> #/
        DEBUG_ANNOTATE_SIDETEXT = 'dst'


LuxMatchConfigs_Default = {
    "mapType": Constants.MAP_TYPES.RANDOM,
    "storeReplay": True,
    "seed": None,
    "debug": False,
    "debugDelay": 500,
    "runProfiler": False,
    "compressReplay": False,
    "debugAnnotations": False,
    "statefulReplay": False,
    "parameters": GAME_CONSTANTS["PARAMETERS"],
}