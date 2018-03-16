#!/usr/bin/env python
from __future__ import division
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
import pickle
import time
class Network():


	def __init__(self, environment_name):

		print("initializing linear Q-network w/ %s" %environment_name)
		self.env = gym.make(environment_name)
		self.environment_name = environment_name
		self.num_actions = self.env.action_space.n
		self.state =  self.env.observation_space.shape[0]
		self.weights =  np.random.uniform(-.1, .1, (self.state, self.num_actions))#np.random.uniform(-.5, .5,(self.state,self.num_actions))
		self.bias = np.zeros(self.num_actions)

	def save_weights(self, suffix = None):
		np.save("weights-%s"%self.environment_name,self.weights)


	def load_model(self, weights_file = "weights.npy"): #??? which model the qnetwork or the agent?
		# Helper function to load an existing model
		self.weights =  np.load(weights_file)
		print("loading model sucessful. model: %s" %self.weights)

	def load_weights(self,weight_file = "weights_bias"):
		# Helper funciton to load model weights. 
		weights_bias = np.load(weight_file)
		weights = weights_bias[:,0] #do we set the model's weights or just load them? 
		bias = weights_bias[:,1]
		self.weights = weights
		self.bias = bias


	def compute_q_values(self, state): 
		actions = np.dot((self.weights).T,state)
		return actions

class DQN_Agent():

	def __init__(self, environment_name, render=False):

		print("initializing DQN agent with environment %s" %environment_name)

		self.network = Network(environment_name)
		self.environment_name = environment_name
		self.render = render
		 
	def epsilon_greedy_policy(self, q_values, env, epsilon= 0.05): # I assume that we fix a state here, so q_values is len(env.action_space.n)
		# Creating epsilon greedy probabilities to sample from.
		policy = np.random.choice(2, p=[epsilon, (1-epsilon)])
		if policy ==0: # explore (choose random action)
			#print("epsilon greedy policy followed")
			action = env.action_space.sample() #randomly select an action from ({0,1} or {0,1,2})
		else: # exploit (greedy policy)-
			#print("not folliwng grreedy")
			action = np.argmax(q_values)

		return action

	def greedy_policy(self, q_values): #q_values refers to a state action pair right?
		# Creating greedy policy for test time. 
		action = np.argmax(q_values)
		return action 

	def train(self, lr = 0.0001, gamma = 1, episodes = 3000, iterations = 1000000, eps= 0.05, memory = False, save = False):
		epsilon = eps
		learning_rate = lr
		epsilon_step_size = 0#(eps)/100000
		env = gym.make(self.environment_name) # directly interface with environmnet
		
		if not memory: 
			i= 0   
			deathsteps = [10]
			episode_count = 0 #
			while(True): #loop through one episode 
				if (i%10000 ==0): # i only matters for iterations ( cartpole environment)
					print("iteration number %i"%i)
					print(self.network.weights)
				
				state = env.reset() #reset game 
				terminal = False
				stepstilldeath = 0
				
				while(True): #simulate until reaching a terminal state

					Q_value = self.network.compute_q_values(state) #compute Q-value
					action = self.epsilon_greedy_policy(Q_value, env, epsilon) # compute epsilon-greedy policy
					if (epsilon > 0.05):
						epsilon -= epsilon_step_size
						assert epsilon >=0.05

					(S_prime, reward, is_terminal, _) = env.step(action)

					if(S_prime[1]==0.5):
						print("goal reached!")

					stepstilldeath+=1
					if(i>iterations and self.environment_name=="CartPole-v0"):
						if(save):
							self.network.save_weights()
						return
					if is_terminal:
						#print(stepstilldeath)
						deathsteps.append(stepstilldeath)
						break
					else:
						Q_prime_max = np.max(self.network.compute_q_values(S_prime))
						if (i%10000 ==0):
							print(" iteration number %i" %i)

							print(np.linalg.norm(Q_value * self.network.weights))
							print(np.mean(deathsteps[-10:]))
						grad =   Q_value * self.network.weights 
						CART_GRAD_THRESH = 1
						TDerr = reward + gamma*Q_prime_max - Q_value[action] 
						self.network.weights += learning_rate*(TDerr)* grad
						if (learning_rate > 0.0001):
							pass
	 					state = S_prime #set state to next 
 					i+=1
	 			if(episode_count ==episodes and self.environment_name=="MountainCar-v0"):
	 				return
				episode_count+=1
		else: 
			pass

	def test(self, model_file=None):
		env = gym.make(self.environment_name)
		stepavg = []
		for i in range(100):
			state = env.reset()
			steps = 0
			while(True): 
				steps+=1
				env.render()
				Q_values = self.network.compute_q_values(state)
				action = self.epsilon_greedy_policy(Q_values,env, epsilon = 0)
				(S_prime, reward, is_terminal, _) = env.step(action)
				if(is_terminal):
					print(steps)
					stepavg.append(steps)
					break
				else:
					state = S_prime
					#time.sleep(0.05)
		print((stepavg))
			
def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str, default = "CartPole-v0")
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--model',dest='model_file',type=str)
	parser.add_argument('--train', dest = 'train', type = bool, default = False)
	parser.add_argument('--num_iters', dest = "num_iters", type = int, default = 1000000)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	environment_name = args.env

	np.random.seed(42)
	Agent = DQN_Agent(environment_name, args.render)
	if(args.train):
		Agent.train(iterations = args.num_iters, save = True)
	else:
		print(Agent.network.weights)
		Agent.test()

if __name__ == '__main__':
	main(sys.argv)
