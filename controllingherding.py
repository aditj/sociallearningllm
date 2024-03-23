import numpy as np
import tqdm as tqdm
import pandas as pd
import matplotlib.pyplot as plt

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


obs_dist_numpy = np.load("parameters/obs_probs.npy")
obs_dist = np.zeros((X,O))

obs_dist[0,0] = obs_dist_numpy[0,::2].sum() 
obs_dist[0,1] = obs_dist_numpy[0,1::2].sum()
obs_dist[1,0] = obs_dist_numpy[1:,::2].sum()
obs_dist[1,1] = obs_dist_numpy[1:,1::2].sum()
obs_dist = obs_dist/obs_dist.sum(axis=1)[:,None]
print(obs_dist)
#exit()
# obs_dist = np.random.rand(X,O)
# obs_dist = obs_dist/obs_dist.sum(axis=1)[:,None]
# obs_dist = perturbed_identity(X,0.5)
obs_dist = np.array([
    [0.3,0.7],
    [0.7,0.3]
])
assert obs_dist.shape == (X, O)

### utility matrix
utility = np.random.rand(X,A)
utility = perturbed_identity(X,1)
utility = np.array([
    [0.5,0],
    [0,1],
])
# utility = np.identity(X)*1
print(utility)
assert utility.shape == (X, A)

data = pd.read_csv("./data/output_csv_cleaned.csv")


grid_size =20
public_belief_grid = np.linspace(0, 1, grid_size)
time_range = np.arange(100)

N_MC =10
thresholds = np.linspace(0,1,grid_size)
public_belief_grid = np.linspace(0, 1, grid_size)
RUN_EXP =0
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
                    
                    obs = int(data[data['state']==state].sample(1).values[0][1:-1].sum()>0)

                    # obs_thres = 0
                    # if np.random.uniform() < obs_thres:
                    #     obs = 1 - obs
                    ### observation belief
                    obs_prob = np.diag(obs_dist[:,obs])

                    if public_belief[0] < threshold:

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

    np.save("parameters/actions_taken_herding.npy",actions_taken)

actions_taken = np.load("parameters/actions_taken_herding.npy")
### meshgrid for public belief and thresholds
mesh = np.meshgrid(public_belief_grid,thresholds)
print(actions_taken.shape)


import seaborn as sns
sns.axes_style("darkgrid")
plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
ax = sns.heatmap(actions_taken.mean(0)[:,:,-20:].mean(axis=2).T,cbar_kws={"label":"Percent Comments Flagged Hate"})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
plt.xticks(range(0,grid_size,4),np.round(public_belief_grid[::4],2),rotation=0)
plt.yticks(range(0,grid_size,4),np.round(public_belief_grid[::4],2),rotation=0)
plt.xlabel("Initial Prior Probability for state 0 (not-HSP)",fontdict={"size":18})
plt.ylabel("Threshold Parameter $\gamma$ of Control Policy",fontdict={"size":18})
plt.text(12,4,"Herding",color = "white",fontsize=20)
plt.savefig("plots/public_belief_thresholds.png")


plt.figure()
# print(actions_taken.mean(0)[0,0,:])
# print(actions_taken.mean(0)[0,2,:])

# print(actions_taken.mean(0)[0,-1,:])

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
prior_idx = [0,10,19]
threshold_idx = [0,10,19]
for i in prior_idx:
    for j in threshold_idx:
        plt.plot(time_range,actions_taken.mean(0)[i,j,:],label=f"Initial Prior: {public_belief_grid[i]}, Threshold: {thresholds[j]}")
plt.legend()
plt.savefig("plots/thresholds.png")