total_epoch=1200           # the total num of msg
interval=0.1               # msg send interval
test_start_time=30         # network fail time (s)
test_duration=30           # network fail duration (s)

tmp_file="tmp123"          # tmp file to tranfer data, will be remove automatically
output_file="my_test.log"  # the python script will write its output into this file

nohup python test_parallel_socket.py -t $total_epoch -i $interval -f $tmp_file 1>$output_file 2>&1 &

ip="0.0.0.0"
while true
do
    if [ -f $tmp_file ]; then
        ip=`cat $tmp_file`
        break
    fi
    sleep 0.1
done
echo "get ip: $ip"

rm $tmp_file

sleep $test_start_time
echo "Network shutsown . . ."
sudo iptables -A INPUT -p tcp -s $ip --dport 50516 -j DROP

sleep $test_duration
sudo iptables -D INPUT -p tcp -s $ip --dport 50516 -j DROP
echo "Network recovered . . ."


