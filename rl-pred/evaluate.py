import keras
from keras.models import load_model
from agent import Agent
from functions import *
import sys

if len(sys.argv) != 3:
    print("Usage: python evaluate.py [vm] [model]")
    exit()

vm_name, model_name = sys.argv[1], sys.argv[2]  # csv file and model name, default null
model = load_model("models/" + model_name)  # loads model
window_size = model.layers[0].input.shape.as_list()[1]  # input shape
agent = Agent(window_size, True, model_name)
data = getDataVec(vm_name,'test')
l = len(data) - 1
batch_size = 32
state = getState(data, 0, window_size + 1)
wrong=0
for t in range(l):
	action = agent.act(state)
	# sit
	load_diff = data[t] - data[t - 1]
	next_state = getState(data, t + 1, window_size + 1)
	reward = 0
	if action == 1:  # higher
		if load_diff > 0:
			print("Higher")
			reward = 1
			wrong-=1

	elif action == 2:  # lower
		if load_diff < 0:
			print("Lower")
			reward = 1
			wrong-=1
	wrong+=1
	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state
	if done:
		print("--------------------------------")
		print("Done with accuracy",float(1-(wrong/l)))
		print("--------------------------------")
