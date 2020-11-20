league.player
=======================

player
----------------------

Player
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.league.player.Player
    :members: __init__

HistoricalPlayer
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.league.player.HistoricalPlayer
    :members: __init__

ActivePlayer
~~~~~~~~~~~~~~~~
.. autoclass:: nervex.league.player.ActivePlayer
    :members: __init__, is_trained_enough, snapshot, mutate, get_job

BattleActivePlayer
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.league.player.BattleActivePlayer
    :members: __init__, is_trained_enough, snapshot, mutate, get_job

SoloActivePlayer
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.league.player.SoloActivePlayer
    :members: __init__, is_trained_enough, snapshot, mutate, get_job

register_player
~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.league.player.register_player

create_player
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.league.player.create_player


starcraft_player
------------------

MainPlayer
~~~~~~~~~~~~~~~
.. autoclass:: nervex.league.starcraft_player.MainPlayer
    :members: __init__, is_trained_enough, snapshot, mutate, get_job

MainExploiter
~~~~~~~~~~~~~~~
.. autoclass:: nervex.league.starcraft_player.MainExploiter
    :members: __init__, is_trained_enough, snapshot, mutate, get_job

LeagueExploiter
~~~~~~~~~~~~~~~
.. autoclass:: nervex.league.starcraft_player.LeagueExploiter
    :members: __init__, is_trained_enough, snapshot, mutate, get_job