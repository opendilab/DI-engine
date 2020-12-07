from .base_league_manager import BaseLeagueManager, create_league, register_league
from .solo_league_manager import SoloLeagueManager
from .player import Player, ActivePlayer, HistoricalPlayer, SoloActivePlayer, BattleActivePlayer, \
    register_player, create_player
from .starcraft_player import MainPlayer, MainExploiter, LeagueExploiter
from .shared_payoff import create_payoff
