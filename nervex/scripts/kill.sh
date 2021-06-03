ps -ef | grep  'nervex' | grep -v grep | awk '{print $2}'|xargs kill -9
