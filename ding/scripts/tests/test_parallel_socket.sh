total_epoch=1200            # the total num of msg
interval=0.1                # msg send interval
size=16                     # data size (MB)
test_start_time=20          # network fail time (s)
test_duration=40            # network fail duration (s)
output_file="my_test.log"   # the python script will write its output into this file
ip="0.0.0.0"

rm -f pytmp_*

nohup python test_parallel_socket.py -t $total_epoch -i $interval -s $size 1>$output_file 2>&1 &

flag=true
while $flag
do
    for file in `ls`
    do
        if [[ $file =~ "pytmp" ]]; then
            ip=`cat $file`
            flag=false
            break
        fi
    done
    sleep 0.1
done
echo "get ip: $ip"

sleep $test_start_time
echo "Network shutsown . . ."
sudo iptables -A INPUT -p tcp -s $ip --dport 50516 -j DROP

sleep $test_duration
sudo iptables -D INPUT -p tcp -s $ip --dport 50516 -j DROP
echo "Network recovered . . ."


