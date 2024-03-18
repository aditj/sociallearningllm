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

X =  flag_state ### *intensity_state*type_state ### number of states
A = X ### number of actions
O = X ### number of observations
P = np.identity(X) ### transition matrix
P = perturbed_identity(X,0)
# P = np.array([[0.5,0,0.5,0],
#      [0,0.5,0,0.5],
#      [0.5,0,0.5,0],
#         [0,0.5,0,0.5]])
print(P)


obs_dist = np.random.rand(X,O)
obs_dist = obs_dist/obs_dist.sum(axis=1)[:,None]
obs_dist = perturbed_identity(X,0.5)
obs_dist = np.array([
    [0.7,0.3],
    [0.3,0.7]
])
assert obs_dist.shape == (X, O)

### utility matrix
utility = np.random.rand(X,A)
utility = perturbed_identity(X,5)
utility = np.array([
    [1,0.5],
    [0.5,1],
])
# utility = np.identity(X)*1
print(utility)
assert utility.shape == (X, A)



grid_size = 20
public_belief_grid = np.linspace(0, 1, grid_size)
time_range = np.arange(100)
N_MC = 100

actions_taken = np.zeros((N_MC,grid_size,time_range[-1]+1))
state_trajectory = np.zeros((N_MC,grid_size,time_range[-1]+1))
observations = np.zeros((N_MC,grid_size,time_range[-1]+1))
RUN_EXP = 0
if RUN_EXP:
    for mc in tqdm.tqdm(range(N_MC)):
        for i, public_belief_0 in enumerate(public_belief_grid):
            ### reset public belief
            public_belief = np.zeros((X,1))
            public_belief[0] = public_belief_0
            public_belief[1] =(1-public_belief_0)

            assert (public_belief.sum()-1) < 1e-10
            state = 1
            for t in time_range:
                state_trajectory[mc,i,t] = state
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
                actions_taken[mc,i,t] = action
                observations[mc,i,t] = obs
        np.save("actions_taken_controlling.npy",actions_taken)
actions_taken = np.load("actions_taken_controlling.npy")
zero_counts = actions_taken[:,:,-10:].mean(axis=2).mean(axis=0)
print(actions_taken[:,:,-10:])
import matplotlib.pyplot as plt
plt.plot(public_belief_grid, zero_counts)
plt.savefig("public_belief_grid_controlling.png")

threshold_policy = lambda x,threshold: 0 if x < threshold else 1
N_MC = 50
thresholds = np.linspace(0,1,50)
public_belief_grid = np.linspace(0, 1, 50)
RUN_EXP = 0
actions_taken = np.zeros((N_MC,len(public_belief_grid),len(thresholds),time_range[-1]+1))
if RUN_EXP:
    for mc in tqdm.tqdm(range(N_MC)):
        for i, public_belief_0 in tqdm.tqdm(enumerate(public_belief_grid)):
            for threshold in thresholds:
                ### reset public belief
                public_belief = np.zeros((X,1))
                public_belief[0] = public_belief_0
                public_belief[1] = 1-public_belief_0

                state = 1
                for t in time_range:
                    ### sample an observation
                    obs = np.random.choice(O, p=obs_dist[state,:])
                    ### observation belief
                    obs_prob = np.diag(obs_dist[:,obs])

                    if public_belief[0] > threshold:
                        action = obs
                    else:
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
                    actions_taken[mc,i,thresholds.tolist().index(threshold),t] = action

    np.save("actions_taken_herding.npy",actions_taken)

actions_taken = np.load("actions_taken_herding.npy")
### meshgrid for public belief and thresholds
mesh = np.meshgrid(public_belief_grid,thresholds)


import seaborn as sns
print(actions_taken[0,22,22,:])
plt.figure()
sns.heatmap(actions_taken[0][:,:,:].sum(axis=2))
plt.xlabel("Public Belief")
plt.ylabel("Threshold")
plt.savefig("public_belief_thresholds.png")