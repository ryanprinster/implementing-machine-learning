import random


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
	if a == 'hit':
		num, color = draw_card()
		player_sum += num*color

		if player_sum > 21 or player_sum < 1: # player busts
			r = -1

	elif a == 'stick':
		dealer_sum = dealers_turn(dealer_sum)

		if dealer_sum > 21 or player_sum > dealer_sum:
			r = 1
		elif dealer_sum == player_sum:
			r = 0
		else: # dealer_sum > player_sum:
			r = -1

	return (dealer_sum, player_sum), r

def pi(s, Q, e):
	''' 
		acts with argmax(Q), epsilon-soft 
	'''
	# init shouldn't matter?
	if (s, 'hit') not in Q:
		Q[(s, 'hit')] = .5 
	if (s, 'stick') not in Q:
		Q[(s, 'stick')] = .5 


	# argmax q
	a_q = 'hit' if Q[(s, 'hit')] > Q[(s, 'stick')] else 'stick'

	#e-soft
	if random.uniform(0, 1) < e:
		return 'hit' if random.uniform(0, 1) < .5 else 'stick'
	else:
		return a_q


def mc_control():
	episodes = 30000
	N = {}
	N_0 = 100.0
	Q = {} 
	e = 1

	# number of episode
	for k in range(episodes):
		# init state and rewards
		s = draw_first_cards()
		r = 0
		traj = []
		
		# play one episode
		while r == 0:
			if (s, 'hit') not in N:
				N[(s, 'hit')] = 0
			if (s, 'stick') not in N:
				N[(s, 'stick')] = 0 

			e = N_0 / (N_0 + N[(s, 'hit')] +  N[(s, 'stick')])
			a = pi(s, Q, e)

			s_next, r = step(s, a)
			traj.append((s,a,r))
			s = s_next

		# update for each s, a in episode 
		for sar in traj:
			s, a, r = sar

			# Update N
			if (s, a) not in N:
				N[(s, a)] = 0
			N[(s, a)] += 1

			# Update Q
			if (s, a) not in Q:
				Q[(s, a)] = r
			Q[(s, a)] += (1.0/N[(s,a)]) * (r - Q[(s, a)])

		# update epsilon
		e = 1.0/(k+1)

		#logging
		if k % 1000 == 0:
			print("episode: ", k)

	return Q, N

Q, N = mc_control()	
s_a = ((10, 20), 'stick')
s_a_ = ((10, 20), 'hit')
print(Q[s_a], N[s_a])
print(Q[s_a_], N[s_a_])

