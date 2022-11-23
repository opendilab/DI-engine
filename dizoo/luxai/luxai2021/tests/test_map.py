import time
from unittest import TestCase
import json
import os
from luxai2021.game.actions import MoveAction
from ..game.constants import Constants
from ..game.game import Game
from ..game.game_constants import GAME_CONSTANTS

class TestMap(TestCase):
    def test_gen_game(self):
        print("Testing generating game...")
        LuxMatchConfigs = {
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

        game = Game(LuxMatchConfigs)

        # Print the game map
        print(game.map.get_map_string())
        print("Map shape: %i,%i" % (len(game.map.map), len(game.map.map[0])))
        assert len(game.map.map) >= 5
        assert len(game.map.map[0]) >= 5
        assert len(game.cities) == 2

        # Print game stats
        print(game.stats)
        assert game.stats["teamStats"][0]["workersBuilt"] == 1

        # Print game state
        print(game.state)
        assert len(game.state["teamStates"][0]["units"]) == 1

        print("Passed game creation test!")
        return True

    def test_gen_game_seed(self):
        print("Testing generating specific game...")
        LuxMatchConfigs = {
            "mapType": Constants.MAP_TYPES.RANDOM,
            "storeReplay": True,
            "seed": 123456789,
            "debug": False,
            "debugDelay": 500,
            "runProfiler": False,
            "compressReplay": False,
            "debugAnnotations": False,
            "statefulReplay": False,
            "parameters": GAME_CONSTANTS["PARAMETERS"],
        }

        game = Game(LuxMatchConfigs)

        # Print the game map
        print("Map for seed 123456789:")
        print(game.map.get_map_string())

        # Test units
        units = list(game.get_teams_units(Constants.TEAM.A).values())
        assert len(units) == 1

        # Try moving a unit, not a great test since maybe can't move North and
        # opponent may be beside this unit.
        test = {}
        unit = units[0]
        oldCellPosition = game.map.get_cell_by_pos(unit.pos)
        newCellPosition = game.map.get_cell_by_pos(
            unit.pos.translate(Constants.DIRECTIONS.NORTH, 1)
        )
        action = MoveAction(
            Constants.TEAM.A,
            unit.id,
            Constants.DIRECTIONS.NORTH
        )

        # Move the unit and run a single turn
        assert len(oldCellPosition.units) == 1
        assert len(newCellPosition.units) == 0
        print(unit.cargo)
        assert unit.cargo[Constants.RESOURCE_TYPES.WOOD] == 0

        gameOver = game.run_turn_with_actions([action])

        print(game.map.get_map_string())
        assert gameOver == False
        assert len(oldCellPosition.units) == 0
        assert len(newCellPosition.units) == 1
        print(unit.cargo)
        assert unit.cargo[Constants.RESOURCE_TYPES.WOOD] == 60

        # Let the game run it's course
        while not gameOver:
            gameOver = game.run_turn_with_actions([])
        print(game.map.get_map_string())

        return True

    def test_map_gen_valid(self):
        print("Testing game map validity against 100 seeds")
        with open(f"{os.path.dirname(__file__)}/testmaps.json", "r") as f:
            all_map_gt = json.load(f)
        for seed in all_map_gt.keys():
            # if int(seed) < 10000: continue
            LuxMatchConfigs = {
                "seed": seed,
            }

            game = Game(LuxMatchConfigs)
            
            map_gt = all_map_gt[seed]
            try:
                assert len(map_gt) == game.map.height
                assert len(map_gt[0]) == game.map.width
            except:
                print(f"Map dimensions mismatch. Seed {seed}. Groundtruth {len(map_gt)} x {len(map_gt[0])}. Generated {game.map.height} x {game.map.width}")
                assert False
            for x in range(game.map.width):
                for y in range(game.map.height):
                    cell = game.map.get_cell(x, y)
                    try:
                        if cell.has_resource():
                            assert cell.resource.amount == map_gt[y][x]["resource"]["amount"]
                            assert cell.resource.type == map_gt[y][x]["resource"]["type"]
                        else:
                            assert map_gt[y][x]["resource"] == None
                        assert cell.is_city_tile() == map_gt[y][x]["citytile"]
                    except:
                        print(f"Map mismatch at ({x}, {y}). Seed {seed}")
                        print("Groundtruth", map_gt[y][x])
                        resource_info = None
                        if cell.has_resource():
                            resource_info = {"resource": {"type": cell.resource.type, "amount": cell.resource.amount}}
                        print("Generated", resource_info, f"Is CT: {cell.is_city_tile()}")
                        assert False
        return True

    def test_gen_game_seed(self):
        print("Testing game simulation speed")
        LuxMatchConfigs = {
            "seed": 123456789,
        }

        game = Game(LuxMatchConfigs)

        # Play 10 games to measure performance
        start_time = time.time()
        for i in range(10):
            gameOver = False
            while not gameOver:
                gameOver = game.run_turn_with_actions([])
            game.reset()
        total_time = time.time() - start_time

        print("Simple empty game: %.3f seconds per full game." % (total_time / 10.0))
        assert (total_time / 10.0) <= 2.0  # Normally takes ~0.312 seconds per game on my device

        return True
