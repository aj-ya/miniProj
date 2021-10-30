from agent import Agent
from functions import *
import sys

if len(sys.argv) != 4:
	print("Usage: python train.py [vm] [window] [episodes]")
	exit()

vm_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(window_size)
data = getDataVec(vm_name,'train')
l = len(data) - 1
batch_size = 32
for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    wrong=0
    print(l)
    for t in range(l):
        action = agent.act(state)
        reward = 0
        load_diff=data[t]-data[t-1]
        next_state = getState(data, t + 1, window_size + 1)
        if action == 1: #higher
            if(load_diff>0):
                print("Higher")
                reward=1
                wrong-=1
		
        elif action == 2: #lower
            if(load_diff<0):
                print("Lower")
                reward=1
                wrong-=1
        wrong+=1
        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
        	print("--------------------------------")
        	print("Done with accuracy->",float(1-(wrong/(l+1))))
        	print("--------------------------------")

        if len(agent.memory) > batch_size:
        	agent.expReplay(batch_size)

    if e % 10 == 0:
    	agent.model.save("models/model_ep" + str(e))
