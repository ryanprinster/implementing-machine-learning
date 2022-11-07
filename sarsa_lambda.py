import random
from collections import defaultdict

def enumerate_states():
	return [(dealer, player) for dealer in range(1, 10 + 1) for player in range(1, 21 + 1)]

def enumerate_actions():
	return ['hit', 'stick']

def draw_card():
	num = random.randint(1, 10)
	color = 1 if random.uniform(0, 1) > 1.0/3 else -1
	return num, color

def draw_first_cards():
	#first card is always black
	dealer_sum, player_sum = random.randint(1, 10), random.randint(1, 10)
	return dealer_sum, player_sum

def dealers_turn(dealer_val):
	while dealer_val < 17:
		num, color = draw_card()
		dealer_val += num*color
	# can the dealer bust going under 1?
	return dealer_val

def step(s, a):

	dealer_sum, player_sum = s
	r = 0
	is_terminal_state = False
	if a == 'hit':
		num, color = draw_card()
		player_sum += num*color

		if player_sum > 21 or player_sum < 1: 
			# player busts
			r = -1
			is_terminal_state = True

	elif a == 'stick':
		dealer_sum = dealers_turn(dealer_sum)

		if dealer_sum > 21 or player_sum > dealer_sum: 
			# dealer busts or player beats dealer
			r = 1
			is_terminal_state = True
		elif dealer_sum == player_sum:
			# dealer and player tie
			r = 0
			is_terminal_state = True
		else: 
			# dealer beats player
			r = -1
			is_terminal_state = True

	return (dealer_sum, player_sum), r, is_terminal_state

def pi(s, Q, e):
	''' 
		acts with argmax(Q), epsilon-soft 
	'''
	print("s: ", s)
	assert(s[0] >= 1 and s[0] <= 10)
	assert(s[1] >= 1 and s[1] <= 21)

	Q.setdefault((s, 'hit'), .5)
	Q.setdefault((s, 'stick'), .5)

	# argmax q
	a_q = 'hit' if Q[(s, 'hit')] > Q[(s, 'stick')] else 'stick'

	#e-soft
	if random.uniform(0, 1) < e:
		return 'hit' if random.uniform(0, 1) < .5 else 'stick'
	else:
		return a_q

def mse(Q_pred, Q_true):
	s_a_pairs = [(s,a) for s in enumerate_states() for a in enumerate_actions()]
	mse = 0
	for s_a in s_a_pairs:
		mse += (Q_true[s_a] - Q_pred[s_a])**2
	return mse


def mc_control():
	episodes = 30000
	N = defaultdict(int)
	N_0 = 100.0
	Q = defaultdict(int)
	epsilon = 1 
	gamma = .95

	# number of episode
	for k in range(episodes):
		# init state and rewards
		s = draw_first_cards()
		r = 0
		traj = []
		is_terminal_state = False

		print("playing episode: ", k)
		
		# play one episode
		while not is_terminal_state:
			epsilon = N_0 / (N_0 + N[(s, 'hit')] +  N[(s, 'stick')])
			a = pi(s, Q, epsilon)
			s_next, r, is_terminal_state = step(s, a)	

			traj.append((s,a,r))
			s = s_next

			if is_terminal_state: 
				break

		# update for each s, a in episode 
		print("traj: ", traj)
		for t, sar in enumerate(traj):
			s, a, r = sar

			# Update N
			N[(s, a)] += 1

			# Update Q
			Q[(s, a)] += (1.0/N[(s,a)]) * (r - Q[(s, a)])

		#logging
		if k % 1000 == 0:
			print("episode: ", k)

	return Q, N

def sarsa_lambda_control(lambda_=.8, Q_true={}):
	# Hyperparams
	episodes = 8000
	gamma = .99
	alpha = .01

	# Initializations
	N, N_0 = defaultdict(int), 100.0
	Q = defaultdict(int)
	epsilon = .1

	# Repeat (for each episode)
	for k in range(episodes):
		E = defaultdict(int)

		# init state and rewards
		s = draw_first_cards()
		r = 0
		
		# Repeat (for each step of episode)
		while r == 0:

			# Take action A, observe R, S'
			# epsilon = N_0 / (N_0 + N[(s, 'hit')] +  N[(s, 'stick')])
			a = pi(s, Q, epsilon)
			s_next, r = step(s, a)

			
			if r != 0: # action was terminal
				delta = r
			else:
				# Choose A' from S' using policy derived from Q (e.g. epsilon-greedy)
				a_prime = pi(s_next, Q, epsilon)

				# delta = r + gamma * Q[(s_next, a_prime)] - Q[(s, a)]
				delta = gamma * Q[(s_next, a_prime)] - Q[(s, a)]

			# Update eligiblity trace
			E[(s, a)] += 1

			# Update the whole state space
			for s_ in enumerate_states(): #enumerate state space
				for a_ in enumerate_actions():
					# if E[(s_, a_)] != 0: 
					# 	print("alpha, ", alpha)
					# 	print("delta, ", delta)
					# 	print("E[(s_, a_)], ", E[(s_, a_)])
					Q[(s_, a_)] += alpha * delta * E[(s_, a_)]
					E[(s_, a_)] = (gamma * lambda_ ) * E[(s_, a_)]

			# add terminal state
			if r != 0: #action was terminal
				break

			s, a = s_next, a_prime

		# update epsilon
		epsilon = 1.0/(k+1)

		#logging
		if k % 100 == 0:
			print("episode: ", k)
			print("mse: ", mse(Q, Q_true))


	return Q, N

# Q_true, _ = mc_control()	
from pprint import pprint


Q_true, _ = mc_control()

pprint(Q_true)

# Q_pred, _ = sarsa_lambda_control(Q_true=Q_true)	

# pprint(Q_pred)


