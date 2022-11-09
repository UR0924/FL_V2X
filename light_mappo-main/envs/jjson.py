import json
import scipy
import scipy.io
import numpy as np
import os
label = 'GRU'
s = json.load(open('rewards.json','r',encoding='utf-8'))
steps=[]
smooth=[]
for i in range(len(s)):
    smooth.append(s[i][0])
    steps.append(s[i][-2])
print(smooth)


current_dir = os.path.dirname(os.path.realpath(__file__))
smooth_rewards = np.asarray(smooth)
reward_path = os.path.join(current_dir, "result/" + label + '/smooth_rewards.mat')
if not os.path.exists(os.path.dirname(reward_path)):
    os.makedirs(os.path.dirname(reward_path))
scipy.io.savemat(reward_path, {'smooth_rewards': smooth_rewards})