ps -ef | grep  'ding' | grep -v grep | awk '{print $2}'|xargs kill -9
