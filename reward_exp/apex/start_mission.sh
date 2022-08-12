kubectl delete dijob lxy-pong-dqn-apex-seed-0 -n di
kubectl delete dijob lxy-pong-dqn-apex-seed-1 -n di
kubectl delete dijob lxy-pong-dqn-apex-seed-2 -n di
alias render_template='python -c "from jinja2 import Template; import sys; print(Template(sys.stdin.read()).render());"'
cat /mnt/nfs/lixueyan/reward/apex/lxy_pong_dqn_apex.yaml.jinja2 | render_template > /mnt/nfs/lixueyan/reward/apex/lxy_pong_dqn_apex.yaml
cat /mnt/nfs/lixueyan/reward/apex/lxy_pong_dqn_apex.yaml.jinja2 | render_template |kubectl apply -f -
