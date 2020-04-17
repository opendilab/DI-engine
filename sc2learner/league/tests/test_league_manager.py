import pytest
from sc2learner.league.league_manager import LeagueManager


@pytest.mark.unittest
class TestLeagueManager:
    def test(self):
        league_manager = LeagueManager({}, lambda x, y: 0, lambda x, y: 0, lambda x: 0)
        league_manager.run()
        league_manager.close()
