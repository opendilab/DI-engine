league.player
=======================

player
----------------------

Player
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.league.player.Player
    :members: __init__

HistoricalPlayer
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.league.player.HistoricalPlayer
    :members: __init__

ActivePlayer
~~~~~~~~~~~~~~~~
.. autoclass:: ding.league.player.ActivePlayer
    :members: __init__, increment_eval_difficulty

NaiveSpPlayer
~~~~~~~~~~~~~~~~
.. autoclass:: ding.league.player.NaiveSpPlayer
    :members: is_trained_enough, snapshot, mutate, get_job, increment_eval_difficulty

create_player
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: ding.league.player.create_player

MainPlayer
~~~~~~~~~~~~~~~
.. autoclass:: ding.league.starcraft_player.MainPlayer
    :members: is_trained_enough, snapshot, mutate, get_job

MainExploiter
~~~~~~~~~~~~~~~
.. autoclass:: ding.league.starcraft_player.MainExploiter
    :members: is_trained_enough, snapshot, mutate, get_job

LeagueExploiter
~~~~~~~~~~~~~~~
.. autoclass:: ding.league.starcraft_player.LeagueExploiter
    :members: is_trained_enough, snapshot, mutate, get_job
