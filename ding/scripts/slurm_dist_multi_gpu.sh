export PYTHONUNBUFFERED=1
export ENABLE_LINKLINK=true
LH=SH-IDC1-10-198-34-42
CLH=SH-IDC1-10-198-34-34
ding -m dist --module config -p slurm -c qbert_dqn_config.py -s 0 -lh $LH -clh $CLH

srun -p spring_scheduler -w $LH --mpi=pmi2 -n3 --job-name=learner0 --comment=spring-submit --gres=gpu:3 ding -m dist --module learner --module-name learner0 -c qbert_dqn_config.py.pkl -s 0 &
srun -p spring_scheduler -w $CLH --job-name=collector0 --comment=spring-submit --gres=gpu:1 ding -m dist --module collector --module-name collector0 -c qbert_dqn_config.py.pkl -s 0 &
srun -p spring_scheduler -w $CLH --job-name=collector1 --comment=spring-submit --gres=gpu:1 ding -m dist --module collector --module-name collector1 -c qbert_dqn_config.py.pkl -s 0 &
srun -p spring_scheduler -w $LH --job-name=aggregator0 --comment=spring-submit ding -m dist --module learner_aggregator --module-name learner_aggregator0 -c qbert_dqn_config.py.pkl -s 0

ding -m dist --module coordinator -p slurm -c qbert_dqn_config.py.pkl -s 0
