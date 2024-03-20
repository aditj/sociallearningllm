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
intensity_state = 5
type_state = 1
X =  flag_state*intensity_state*type_state ### number of states
A = X ### number of actions
O = 64 ### number of observations
P = np.identity(X) ### transition matrix
P = perturbed_identity(X,0)

### observation distribution
obs_dist = np.load("parameters/obs_probabilities.npy").T
obs_dist = np.concatenate([obs_dist[0,:].reshape(1,-1),np.ones((intensity_state-1,O))*1e-7,obs_dist[1:,:]],axis=0)
obs_dist = obs_dist/obs_dist.sum(axis=1)[:,None]
# obs_dist = np.repeat(perturbed_identity(X,1),O//X,axis=1) 
# obs_dist = np.concatenate([np.zeros((X,1)),np.zeros((X,1)),obs_dist],axis=1)

# obs_dist = obs_dist/obs_dist.sum(axis=1)[:,None]

print(obs_dist)
assert obs_dist.shape == (X, O)

### utility matrix
utility = np.random.rand(X,A)
utility = perturbed_identity(X,0.7)
print(utility)
assert utility.shape == (X, A)


grid_size = 20
public_belief_grid = np.linspace(0, 1, grid_size)
time_range = np.arange(20)
N_MC = 1
Xs = [0,5,6,7,8,9]
len_Xs = len(Xs)
actions_taken = np.zeros((N_MC,grid_size,len_Xs,time_range[-1]+1))

RUN_EXP = 1
if RUN_EXP:
    for mc in tqdm.tqdm(range(N_MC)):
        for j,init_state in enumerate(Xs):

            for i, public_belief_0 in enumerate(public_belief_grid):
                ### reset public belief
                public_belief = np.zeros((X,1))
                public_belief[0] = public_belief_0
        
                public_belief[init_state] =(1-public_belief_0)
                assert (public_belief.sum()-1) < 1e-10
                state = init_state
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
                    
                    # print(t,state,obs,action)
                    actions_taken[mc,i,j,t] = action
            np.save("parameters/actions_taken.npy",actions_taken)
actions_taken = np.load("parameters/actions_taken.npy")
zero_counts = actions_taken[:,:,-10:].mean(axis=3).mean(axis=0)
print(actions_taken[:,:,-10:].mean(axis=3).mean(axis=0))
import seaborn as sns
labels = ["[0,0]","[1,1]","[1,2]","[1,3]","[1,4]","[1,5]"]
sns.axes_style("darkgrid")
import matplotlib.pyplot as plt
plt.plot(public_belief_grid,zero_counts,label=labels )
plt.xlabel("Initial Public Belief $\pi$([0,0])",fontdict={"size":18})
plt.ylabel("Average Action $a$ of last 10 agents",fontdict={"size":18})
plt.yticks(Xs,labels)
plt.legend(title="Underlying State")
plt.savefig("plots/public_belief_grid.png")
