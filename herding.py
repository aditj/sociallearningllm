import numpy as np
import tqdm as tqdm
def p_a_public_belief(a,obs_dist,utility,obs,A,P,public_belief):
    flag = True
    obs_prob = np.diag(obs_dist[:,obs])
    for a_ in range(A):
        flag = flag and utility[:,a].T@obs_prob@P.T@public_belief >= utility[:,a_].T@obs_prob@P.T@public_belief
    return flag

def T(public_belief,a,X,O,obs_dist,P,A,utility):
    R_a_pi = np.identity(X)
    for x in range(X):
        for o in range(O):
            R_a_pi[x,x] += p_a_public_belief(a,obs_dist,utility,o,A,P,public_belief) * obs_dist[x,o]
    return R_a_pi@P.T@public_belief

### perturbed identity matrix
def perturbed_identity(X,epsilon):
    P = np.identity(X) + epsilon*np.random.rand(X,X)
    P = P / P.sum(axis=1)[:,None]
    return P

flag_state = 2
intensity_state = 3
type_state = 2
X =  flag_state*intensity_state*type_state ### number of states
A = X ### number of actions
O = X ### number of observations
P = np.identity(X) ### transition matrix
P = perturbed_identity(X,0.05)
print(P)
state_var = [0,0,0]
state = state_var[0]*intensity_state*type_state + state_var[1]*type_state + state_var[2]

obs_dist = np.random.rand(X,O)
obs_dist = obs_dist/obs_dist.sum(axis=1)[:,None]
obs_dist = perturbed_identity(X,0.02)
print(obs_dist)
assert obs_dist.shape == (X, O)

### utility matrix
utility = np.random.rand(X,A)
utility = np.identity(X)
assert utility.shape == (X, A)


grid_size = 100
public_belief_grid = np.linspace(0, 1, grid_size)
time_range = np.arange(1000)
actions_taken = np.zeros((grid_size,time_range[-1]+1))
state_trajectory = np.zeros((grid_size,time_range[-1]+1))
observations = np.zeros((grid_size,time_range[-1]+1))
RUN_EXP = 1
if RUN_EXP:
    for i, public_belief_0 in tqdm.tqdm(enumerate(public_belief_grid)):
        ### reset public belief
        public_belief = np.zeros((X,1))
        for state_var_1 in range(intensity_state):
            for state_var_2 in range(type_state):
                public_belief[0*intensity_state*type_state + state_var_1*type_state + state_var_2] = public_belief_0/(intensity_state*type_state)
                public_belief[1*intensity_state*type_state + state_var_1*type_state + state_var_2] = (1 - public_belief_0)/(intensity_state*type_state)
        public_belief = public_belief/public_belief.sum()    

        for t in time_range:
            ### sample an observation
            obs = np.random.choice(O, p=obs_dist[state,:])
            ### observation belief
            obs_prob = np.diag(obs_dist[:,obs])
            ### computing private belief
            private_belief = obs_prob @ P.T @ public_belief
            normalization_factor = np.ones((1, O)) @ obs_dist.T @ P.T @ public_belief
            private_belief = private_belief / normalization_factor
            ### action selection
            action = np.argmax(utility.T @ private_belief)
            ### update public belief
            public_belief = T(public_belief,action,X,O,obs_dist,P,A,utility)
            public_belief = public_belief / np.sum(public_belief)
            ### update state
            state = np.random.choice(X, p=P[state,:])
            # print(t, state, obs, public_belief.flatten(),action, sep="\t")
            # print(t,state,obs,action)
            actions_taken[i,t] = action
            state_trajectory[i,t] = state
            observations[i,t] = obs
    np.save("actions_taken.npy",actions_taken)
actions_taken = np.load("actions_taken.npy")
zero_counts = (actions_taken == 0).sum(axis=1)

import matplotlib.pyplot as plt
plt.plot(public_belief_grid, zero_counts)
plt.savefig("public_belief_grid.png")

