# Multi-Agent Particle

This environment is based on the openAI's work <https://github.com/openai/multiagent-particle-envs>, transplanted to DI-engine.

## QTRAN experiment

We also support a modified predator-prey environment as in [QTRAN](https://arxiv.org/abs/1905.05408) paper to evaluate the superiority of QTRAN than QMIX. The predators get a team reward of `10`, if two or more catch a prey at the same time, but they are given negative reward `−P`, when only one predator catches the prey. This setting requires a higher degree of cooperation.

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="image/modified_predator_prey.png">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">Figure 1.  The modified predator-prey environment. Good agents (green) are faster and want to avoid being hit by adversaries (red). Adversaries are slower and want to hit good agents. Obstacles (large black circles) block the way. Positive reward is given only if multiple predators catch a prey simultaneously.</div> </center>

### Penalty

Figure 2 show that as the penalty term `P` increases gradually (from 0 to 2, 5), the reward obtained under the same environment step becomes smaller and smaller. We use QTRAN in this experiment.  The penalty=`-P`.

<center>    <img style="border-radius: 0.3125em;    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"     src="image/qtran_with_different_penalty.png">    <br>    <div style="color:orange; border-bottom: 1px solid #d9d9d9;    display: inline-block;    color: #999;    padding: 2px;">Figure 2. As the penalty term `P` increases gradually, the reward decreases. </div> </center>

### QTRAN vs. QMIX

Qtran achieves better performance than QMIX at the same env step.

![image-20210821235020851](image/qtran_vs_qmix_penalty2.png)

![image-20210821234841919](image/qtran_vs_qmix_penalty5.png)