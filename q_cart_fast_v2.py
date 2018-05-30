# -*- coding: utf-8 -*-
import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from scipy import stats
import pickle
import datetime

# my files
# import os
# import importlib
# import DeepNetPI
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # avoids a warning
# importlib.reload(DeepNetPI)
# from DeepNetPI import TfNetwork

env = gym.make('CartPole-v0')
is_record = False
is_view = True
if is_record:
	env = wrappers.Monitor(env, 'RL_movies/cartpole-experiment-3',force=True, video_callable=lambda episode_id: True)
start_time = datetime.datetime.now()

# --- inputs ---

# NN params
learning_rate = 0.0006 	# 0.0005
hidden_size = 100. 	# no. hidden neurons
n_ensemble = 10			# no. in ensemble
optimiser_in = 'adam' 	# optimiser: adam, SGD
init_stddev = 5.0 		# 5.0 for single or 7.0 for ensemble
init_stddev_2 = 0.18/np.sqrt(hidden_size) # 0.2 0.18 # second layer initialisation dist
# lambda_weight = 0. # 0.00001
# var_data = np.square(10.0) # estimate of data noise variance
# scale_l_a = 1. # 0.0001
# lambda_anchor = [scale_l_a*var_data*1/np.square(init_stddev),
	# scale_l_a*var_data*np.square(1/(3*1/np.sqrt(hidden_size)))] # diff lambdas for each layer

lambda_anchor = [0.000001,0.1] # for both - 0.000001,0.1
# lambda_anchor = [0.000001,1.0] # for ensemble
# lambda_anchor = [0.,0.]

# print('lambda_anchor: ',lambda_anchor)
# equiv_prior_stddev = np.sqrt(var_data / lambda_weight)
# print('equiv_prior_stddev: ',equiv_prior_stddev)

# RL params
state_size = 4 			# inputs from state
n_actions = 2 			# no. possible actions (discrete)
n_episodes = 50		# no. epsiodes to run # 140 1000
max_episode_len = 200 	# no. timesteps              
gamma = 0.97 			# decay of reward
batch_size = 64 		# no. samples to train on per train session
n_burn_in = 0   		# no. episodes to run before training 10
eps_max = 1.			# epsilon greedy max
eps_min = 0. 			# epsilon greedy minimum
eps_decay = 30.  		# episodes until eps reaches eps_min # 60 400
assert eps_decay > n_burn_in # must be larger
n_train_reps = 2 # 10	# no. times to repeat training for each batch
n_large = 0 			# no. of large variance samples to select from last episode, 0. to turn off
rand_focus_select = False # whether to choose focus samples randomly
force_e_greedy = True 	# force ensemble to sample via epsilon greedy
force_greedy_decayed = False # force ensemble to use full greedy behaviour after eps has decayed

# other params
n_runs = 1				# no. runs to perform
n_steps = 5 			# no. episodes to aggregate into 
view_params = False		# plot param histograms
view_confidence = False	# plot analysis of confidence in actions
view_quality = True		# plot final training progress
# prob_sample = False		# whether to use probabilistic sampling of buffer
save_graphs = False		# whether to save all graphs showed
save_results = False	# save results as pickle
plot_compare = False	# add competing line to plot (load from pickle)
compare_to = '15_42_59.p' # pickle filename to plot compare to
col_comp = 'b'			# colour for comparison


# --- utils ---

def print_time(string_in):
	print(string_in, '\t -- ', datetime.datetime.now().strftime('%H:%M:%S'))
	return

def get_time():
	return datetime.datetime.now().strftime('%m_%d_%H_%M_%S')

# --- NN class ---

class NN():
	def __init__(self):
		self.inputs = tf.placeholder(tf.float32, [None, state_size], name='inputs')
		self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
		self.one_hot_actions = tf.one_hot(self.actions_, n_actions)
		self.q_target = tf.placeholder(tf.float32, [None], name='target')
		
		# we use Dense instead of dense - so can access weights more easily
		self.layer_1_w = tf.layers.Dense(hidden_size,
			activation=tf.nn.tanh, #trainable=False,
			kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev),
			bias_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev))
			# kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambda_weight),
			# bias_regularizer=tf.contrib.layers.l2_regularizer(scale=lambda_weight))
		self.layer_1 = self.layer_1_w.apply(self.inputs)

		# self.layer_1_b_w = tf.layers.Dense(hidden_size,
		# 	activation=tf.nn.tanh, #trainable=False,
		# 	kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev),
		# 	bias_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev))
		# self.layer_1_b = self.layer_1_b_w.apply(self.layer_1)


		self.output_w = tf.layers.Dense(n_actions, 
			activation=None, use_bias=True,
			kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_2))
			# kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=lambda_weight))

		self.output = self.output_w.apply(self.layer_1)

		self.q = tf.reduce_sum(tf.multiply(self.output, self.one_hot_actions), axis=1)
		
		if optimiser_in == 'adam':
			self.opt_method = tf.train.AdamOptimizer(learning_rate)
		elif optimiser_in == 'SGD':
			self.opt_method = tf.train.GradientDescentOptimizer(learning_rate)

		self.loss_ = tf.reduce_mean(tf.square(self.q_target - self.q))
		self.optimizer = self.opt_method.minimize(self.loss_)
		return


	def get_weights(self, sess):
		# method to return current params - yes it rly does seem this hard..

		ops = [self.layer_1_w.kernel, self.layer_1_w.bias, self.output_w.kernel]
		w1, b1, w2 = sess.run(ops)
		# b2 = sess.run(self.output_w.bias)
		# print('\nb2: ',b2)
		return w1, b1, w2


	def anchor(self, sess, regularise=False):
		# method to set loss to account for anchoring

		# get weights
		ops = [self.layer_1_w.kernel, self.layer_1_w.bias, self.output_w.kernel]
		w1, b1, w2 = sess.run(ops)

		# to do normal regularisation
		self.w1_init, self.b1_init, self.w2_init = 0.,0.,0. # overwrite for normal regulariser

		# set squared loss around it
		loss_anchor = lambda_anchor[0]*tf.reduce_sum(tf.square(self.w1_init - self.layer_1_w.kernel))
		loss_anchor += lambda_anchor[0]*tf.reduce_sum(tf.square(self.b1_init - self.layer_1_w.bias))
		loss_anchor += lambda_anchor[1]*tf.reduce_sum(tf.square(self.w2_init - self.output_w.kernel))
		
		# combine with original loss
		N = batch_size # should figure out input size within tf really...
		self.loss_ = self.loss_ + loss_anchor/N

		# reset optimiser
		self.optimizer = self.optimizer = self.opt_method.minimize(self.loss_)
		return


	def plot_params(self,sess):
		# histogram of current parameters

		w1,b1,w2 = NNs[0].get_weights(sess)

		fig = plt.figure()
		ax = fig.add_subplot(311)
		ax.hist(w1.ravel(),bins=50)
		ax.set_ylabel('w1')
		ax = fig.add_subplot(312)
		ax.hist(b1.ravel(),bins=50)
		ax.set_ylabel('b1')
		ax = fig.add_subplot(313)
		ax.hist(w2.ravel(),bins=50)
		ax.set_ylabel('w2')
		fig.show()
		return


# --- main code here ---

print_time('started')

tf.reset_default_graph()
runs_reward_list = [] 	# rewards per episode for each run
runs_reward_batch = [] 	# aggregated per x runs
runs_explore_list = [] 	# how many times explored for each run
runs_explore_batch = []	# aggregated per x runs
runs_solved = []
for run in range(0,n_runs):
	print('\n\n --- run ', run, ' of ', n_runs-1, ' ---')

	# create all NNs
	NNs=[]
	for i in range(0,n_ensemble):
		NNs.append(NN())

	env.reset()
	reward_list = []
	explore_list = []
	experience_replay = []
	consecutive_wins = 0
	print_win=True
	# runs_reward = []
	l = 0 # loss
	plot_focus = 0	# which plot to draw for confidence plots

	# initialise NNs
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	# set anchoring regularisation around initialisations for ensemble
	for i in range(0,n_ensemble):
		if n_ensemble > 1:
			NNs[i].anchor(sess, regularise=False) # anchor around init params
		else:
			NNs[i].anchor(sess, regularise=True) # regularise around zero

	# view param distribution
	if view_params:
		NNs[0].plot_params(sess)

	counter = 0
	prob_list=[] # store certainty of actions
	var_action = []
	for episode in range(0, n_episodes):

		ens_i = np.random.randint(0,n_ensemble) # select random NN from ensemble
		total_reward = 0 # per episode
		explore = 0
		observation = env.reset()

		# test if 'solved' - need 100 episodes with avg >= 195
		if len(reward_list)>100 and np.sum(reward_list[-100:])/100 >= 195. and print_win:
			print('\n-- LEVEL 1 RL COMPLETE --')
			print('took ' + str(episode-100) + ' episodes')
			runs_solved.append(episode)
			print_win = False # only print once
			# break
		
		# linear epsilon greedy
		e = min(eps_max, max(eps_min, eps_max - (eps_max - eps_min) * (episode-n_burn_in) / (eps_decay-n_burn_in) ))
		
		# exponential epsilon greedy
		# e = 1.0 * 0.98**(episode/2)
		# e = 0.97**(episode)

		for i in range(max_episode_len):
			counter += 1
			if is_record or is_view:
				env.render() # visualise

			if n_ensemble == 1:
				# sample input w epsilon greedy
				if e > np.random.uniform(0, 1):
					action = np.random.randint(n_actions)
					explore += 1
				else:
					feed = {NNs[ens_i].inputs: observation.reshape((1, *observation.shape))}
					Qs = sess.run(NNs[ens_i].output, feed_dict=feed)
					action = np.argmax(Qs)
				prob_list.append(0) # fill these with dummy variables
				var_action.append(0)
			else:
				# probabilistic selection of action
				Qs_ens = []
				feed = {}
				ops = []
				for j in range(0,n_ensemble):
					feed[NNs[j].inputs] = observation.reshape((1, *observation.shape))
					ops.append(NNs[j].output)
				Qs_ens.append(sess.run(ops, feed_dict=feed))

				Qs_ens = np.array(Qs_ens).squeeze()
				Qs_ens_mean = Qs_ens.mean(axis=0)
				Qs_ens_std = Qs_ens.std(axis=0, ddof=1) # just std dev
				# Qs_ens_std = Qs_ens.std(axis=0, ddof=1)/np.sqrt(n_ensemble) # std ERROR

				# now find prob of action_0 being larger
				# https://math.stackexchange.com/questions/40224/probability-of-a-point-taken-from-a-certain-normal-distribution-will-be-greater
				mu_diff = Qs_ens_mean[0] - Qs_ens_mean[1]
				std_diff = np.sqrt(np.square(Qs_ens_std[0]) + np.square(Qs_ens_std[1]))
				prob_0 = 1 - stats.norm.cdf(-mu_diff/std_diff)

				# select according to that probability
				if np.random.uniform(0, 1) < prob_0:
					action = 0
				else:
					action = 1

				# manually force epsilon greedy behaviour
				if force_e_greedy:
					if e > np.random.uniform(0, 1):
						action = np.random.randint(n_actions)
						explore += 1
					else: # choose greedy behaviour
						if prob_0 > 0.5:
							action = 0
						else:
							action=1

				# manually force fully greedy behaviour
				elif force_greedy_decayed:
					if episode > eps_decay:
						if prob_0 > 0.5:
							action = 0
						else:
							action=1

					else: # normal Thompson criteria
						if prob_0 > 0.5 and action == 1:
							explore += 1
						elif prob_0 < 0.5 and action == 0:
							explore += 1

				# default Thompson uncertainty
				else:
					if prob_0 > 0.5 and action == 1:
						explore += 1
					elif prob_0 < 0.5 and action == 0:
						explore += 1

				# store details about probability for later analysis
				prob_list.append(prob_0)
				var_action.append(Qs_ens_std[action])

			observation_1, reward, done, _ = env.step(action)
			total_reward += reward
			
			# check if finished and store the experience
			if done:
				# print(observation_1) # print interesting obs

				observation_1 = np.zeros(observation.shape)
				if len(experience_replay) > 100000:
					del_i = np.random.randint(0,100000) # delete a random entry from buffer
					del experience_replay[del_i]

				# don't save if last step as don't get proper observation_1
				if i < 199:
					experience_replay.append([observation, action, reward, observation_1, var_action[-1]])
					# only reset if really messed up
					if i < 150 and consecutive_wins<30:
						consecutive_wins = 0
				else:
					consecutive_wins += 1

				# print info
				print('Run: {}'.format(run),
				  'Ep: {}'.format(episode),
				  'Rew: {}'.format(int(total_reward)),
				  'L: {:.3f}'.format(l),
				  'e: {:.3f}'.format(e))
				# env.reset()
				# print('timestep:',i)
				# print([observation, action, reward, observation_1])
				reward_list.append(total_reward)
				explore_list.append(explore)

				# promote the x most uncertain samples from this episode as high priority
				episode_exp = experience_replay[-i:]
				episode_exp = np.array(episode_exp)

				# find indices of x largest variance - these were least certain
				n_large_limit = np.min((n_large,i))
				ind_focus = np.argpartition(episode_exp[:,4], -n_large_limit)[-n_large_limit:]

				# test against uniform random selection
				if rand_focus_select or n_ensemble == 1:
					ind_focus = np.random.randint(0,i,n_large_limit)

				break
				
			else: # if not done
				# add experience to memory
				if len(experience_replay) > 100000:
					del_i = np.random.randint(0,100000)
					del experience_replay[del_i]
				experience_replay.append([observation, action, reward, observation_1, var_action[-1]])
				observation = observation_1
		

			# randomly sample for first x episodes
			if episode > n_burn_in and consecutive_wins < 10: #and episode < eps_decay: # try override for trained model

				# select historic samples to train on
				# choice is much quicker than shuffle
				samples_i = np.random.choice(range(len(experience_replay)),size=batch_size-n_large_limit,replace=True)
				samples=[]

				# recent samples
				for k in ind_focus:
					samples.append(np.array(episode_exp[k]))

				# other samples
				for k in samples_i:
					samples.append(np.array(experience_replay[k]))
				samples = np.array(samples)

				replay_observations = np.vstack(samples[:,0])
				replay_actions = samples[:,1]
				replay_rewards = samples[:,2]
				replay_observations_ = np.vstack(samples[:,3])
				episode_ends = (replay_observations_ == np.zeros(replay_observations[0].shape)).all(axis=1)
		
				# training step - for all NNs in ensemble
				# restructured so quicker - multiple sess calls is slow
				for _ in range(n_train_reps): # repeat training multiple times
					targets = []
					ops = []; feed = {}
					for j in range(0,n_ensemble):
						feed[NNs[j].inputs] = replay_observations_
						ops.append(NNs[j].output)
					target_qs = sess.run(ops, feed_dict=feed)
					target_qs = np.array(target_qs)
					target_qs[:,episode_ends,:] = (0, 0)
					targets = replay_rewards/20. + gamma * np.max(target_qs, axis=2)
					# targets = (replay_rewards-1) + gamma * np.max(target_qs, axis=2)

					# print('\n\ntarget_qs',target_qs)
					# print('targets',targets)

					ops = []; feed = {}
					for j in range(0,n_ensemble):
						feed[NNs[j].inputs] = replay_observations
						feed[NNs[j].q_target] = targets[j]
						feed[NNs[j].actions_] = replay_actions
						ops.append(NNs[j].loss_)
						ops.append(NNs[j].optimizer)
					loss_and_blank = sess.run(ops, feed_dict=feed)
					l = np.mean(loss_and_blank[0::2])  # find mean loss

	# how quality improves per x episodes
	reward_batch = []
	explore_batch = []
	for i in range(0,len(reward_list),n_steps):
		reward_batch.append(sum(reward_list[i:i+n_steps])/n_steps)
		explore_batch.append(sum(explore_list[i:i+n_steps])/n_steps)

	runs_reward_list.append(reward_list)
	runs_explore_list.append(explore_list)
	runs_reward_batch.append(reward_batch)
	runs_explore_batch.append(explore_batch)
	sess.close()

env.close()

# plot how quality improved
if n_ensemble > 1:
	col = 'r'
	col2 = 'mistyrose' # for variance - eps can't do alpha
else:
	col = 'b'
	col2 = 'lightblue'

if view_quality:

	runs_reward_batch = np.array(runs_reward_batch)
	runs_explore_batch = np.array(runs_explore_batch)
	runs_mean = runs_reward_batch.mean(axis=0)
	runs_std = runs_reward_batch.std(axis=0, ddof=1)
	runs_explore_mean = runs_explore_batch.mean(axis=0) / runs_mean * 200# divide by run length
	x_batch = n_steps*np.arange(1,len(runs_reward_batch[0])+1)
	cum_reward_mean = np.cumsum(runs_mean)*n_steps /1000

	fig = plt.figure(figsize=(6, 4))
	ax = fig.add_subplot(111)
	ax.grid()
	ax.fill(np.concatenate([x_batch, x_batch[::-1]]),
	         np.concatenate([runs_mean - 2 * runs_std,
	                        (runs_mean + 2 * runs_std)[::-1]]),
	         alpha=1.0, fc=col2, ec='None', label='2 std devs')

	for i in range(n_runs):
		ax.plot(x_batch,runs_reward_batch[i], col+':', linewidth=1.)
	ax.plot(x_batch,runs_mean, col+'-', linewidth=2.,label=u'Mean reward')
	ax.plot(x_batch,runs_explore_mean, 'g--', linewidth=1.,label=u'Explore rate (x200)')
	ax.set_xlabel('Episodes')
	ax.set_ylabel('Avg. reward for last ' + str(n_steps) + ' epsiodes')
	title = 'n_ensemble ' + str(n_ensemble) + ', hidden_size ' \
		+ str(hidden_size) + ', learning_rate ' + str(learning_rate) \
		+ ', lambda_anc ' + str(lambda_anchor) + '\neps_decay ' + str(eps_decay) \
		+ ', n_train_reps ' + str(n_train_reps) + ', force_e_greedy ' + str(force_e_greedy)
	# ax.set_title(title, fontsize=10)
	# ax.legend()
	runs_solved = np.array(runs_solved)
	ax.plot(runs_solved-100,np.zeros_like(runs_solved)+100,'r+', markersize=11, markeredgecolor='k',markeredgewidth=0.7,label='ep. solved')
	ax.set_ylim([0.,205.])

	if plot_compare:
		runs_reward_batch_comp = pickle.load( open( "RL_outputs/pickles/"+compare_to, "rb" ) )
		runs_mean_comp = runs_reward_batch_comp.mean(axis=0)
		ax.plot(x_batch,runs_mean_comp, col_comp+'-', linewidth=2.,label=u'Comp. Mean')
		# runs_reward_batch_comp = pickle.load( open( "RL_outputs/pickles/"+compare_to, "rb" ) )
		# runs_mean_comp = runs_reward_batch_comp.mean(axis=0)
		# ax.plot(x_batch,runs_mean_comp, col_comp+'-', linewidth=2.,label=u'Comp. Mean')

	ax2 = ax.twinx()
	ax2.plot(x_batch,cum_reward_mean, 'k:', linewidth=2.,label=u'Cum. reward')
	ax2.set_ylabel('Cumulative mean rewards (/1000)')
	ax2.set_ylim(bottom=0.)

	# dummy lines for legend
	ax.plot(x_batch[0],cum_reward_mean[0]*1000, 'k:', linewidth=2.,label=u'Cum. reward')
	ax.plot(x_batch[0],cum_reward_mean[0]*1000, col+':', linewidth=1.,label=u'Single run reward')

	# ax.legend(['Single runs','Mean', 'Explores', '2x Std dev', 'Cum. rewards'])	
	ax.legend(loc='lower right')
	ax.set_xlim([np.min(x_batch),np.max(x_batch)])

	fig.show()

	if save_graphs:
		1==1
		# fig.savefig('RL_outputs/'+get_time()+'_quality.eps', format='eps', dpi=1000, bbox_inches='tight')

if save_results:
	pickle.dump( runs_reward_batch, open( "RL_outputs/pickles/" + get_time() +".p", "wb" ) )


solved_mean = np.average(runs_solved)-100
solved_std = np.std(runs_solved)
solved_se = solved_std / np.sqrt(n_runs)
print('\nEpisodes required before solve, averaged over', n_runs, 'runs:', np.round(solved_mean,2) ,'+/-', 2*np.round(solved_se,2))

# timing info
end_time = datetime.datetime.now()
total_time = end_time - start_time
print('seconds taken:', round(total_time.total_seconds(),1),
	'\nstart_time:', start_time.strftime('%H:%M:%S'), 
	'end_time:', end_time.strftime('%H:%M:%S'))
