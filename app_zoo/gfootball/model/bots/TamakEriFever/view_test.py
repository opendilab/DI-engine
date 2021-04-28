# Set up the Environment.

import time

from kaggle_environments import make

# opponent = "football/idle.py"
# opponent = "football/rulebaseC.py"
opponent = "builtin_ai"

video_title = "chain"
video_path = "videos/" + video_title + "_" + opponent.split("/")[-1].replace(".py",
                                                                             "") + str(int(time.time())) + ".webm"

env = make(
    "football",
    configuration={
        "save_video": True,
        "scenario_name": "11_vs_11_kaggle",
        "running_in_notebook": False
    },
    info={"LiveVideoPath": video_path},
    debug=True
)
output = env.run(["submission.py", opponent])[-1]

scores = [output[i]['observation']['players_raw'][0]['score'][0] for i in range(2)]
print('Left player: score = %s, status = %s, info = %s' % (scores[0], output[0]['status'], output[0]['info']))
print('Right player: score = %s, status = %s, info = %s' % (scores[1], output[1]['status'], output[1]['info']))

env.render(mode="human", width=800, height=600)
