# How to replay a log

1. Set the log path to store episode logs by the following command:

   `env.enable_save_replay('./game_log')`

2. After running the game, you can see some log files in the game_log directory.

3. Execute the following command to replay the log file (*.rcg)

   ` env.replay_log("game_log/20211019011053-base_left_0-vs-base_right_0.rcg")`