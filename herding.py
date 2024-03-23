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

flag_state = 6
intensity_state = 1
type_state = 1
X =  flag_state*intensity_state*type_state ### number of states
A = X ### number of actions
O = 6 ### number of observations
P = np.identity(X) ### transition matrix
P = perturbed_identity(X,0)

### observation distribution
obs_dist_numpy = np.load("parameters/obs_probs.npy")
obs_dist = np.zeros((X,O))
# ### get binary valued vectors for 64
indices = np.arange(2**5)
binary_indices = np.array([list(bin(i)[2:].zfill(6)) for i in indices]).astype(int)
obs_dist[:,0] = obs_dist_numpy[:,::2].sum(axis=1)
obs_dist_numpy = obs_dist_numpy[:,1::2]
for i in range(O):
#     ### filter out indices which have only things less than i 1
    
    obs_dist[:,i] = obs_dist_numpy[:,binary_indices[:,i+1:].sum(axis=1)==0].sum(axis=1)
print(obs_dist)

# obs_dist[:,1] = obs_dist_numpy[:,obs_dist_numpy[:,::2]==0]
# print()
# print(obs_dist[:,1::2][:,::2].sum(axis=1))
# exit()
# obs_dist = np.repeat(perturbed_identity(X,0.2),O//X,axis=1) 
# obs_dist = np.concatenate([np.zeros((X,1)),np.zeros((X,1)),np.zeros((X,1)),np.zeros((X,1)),obs_dist],axis=1)

obs_dist = obs_dist/obs_dist.sum(axis=1)[:,None]

print(obs_dist.shape)

assert obs_dist.shape == (X, O)

### utility matrix
utility = np.random.rand(X,A)
utility = perturbed_identity(X,0.7)
print(utility)
assert utility.shape == (X, A)


grid_size = 100
public_belief_grid = np.linspace(0, 1, grid_size)
time_range = np.arange(10)
N_MC = 50
Xs = [0,1,2,3,4,5]
len_Xs = len(Xs)
actions_taken = np.zeros((N_MC,grid_size,len_Xs,time_range[-1]+1))


data = pd.read_csv("./data/output_csv_cleaned.csv")

RUN_EXP = 0
if RUN_EXP:
    for mc in tqdm.tqdm(range(N_MC)):
        for j,init_state in tqdm.tqdm(enumerate(Xs)):

            for i, public_belief_0 in enumerate(public_belief_grid):
                ### reset public belief
                public_belief = np.zeros((X,1))
                public_belief[0] = public_belief_0
        
                public_belief[init_state] =(1-public_belief_0)
                assert (public_belief.sum()-1) < 1e-10
                state = init_state
                for t in time_range:
                    ### sample an observation
                    obs = data[data['state']==state].sample(1).values[0][1:]
                    obs = np.concatenate([[obs[-1]],obs[:-1]])
                
                    obs = np.max(np.concatenate([np.nonzero(obs==1)[0].flatten(),[0]]))
                    # if state != 0:
                    #     if np.random.uniform() <0.6:
                    #         obs = 0
                        
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
zero_counts = actions_taken[:,:,:,:].mean(axis=3).mean(axis=0)
print(actions_taken[:,:,:,:].mean(axis=3).mean(axis=0))
import seaborn as sns
### use latex for font rendering
## use serif font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
labels = ["0: Not Hate Speech Peddler (HSP)","1: HSP of Intensity 1","2: HSP of Intensity 2","3: HSP of Intensity 3","4: HSP of Intensity 4","5: HSP of Intensity 5"]
labels_ticks = ["0","1","2","3","4","5"]
sns.axes_style("darkgrid")
import matplotlib.pyplot as plt
plt.plot(public_belief_grid,zero_counts,label=labels )
plt.xlabel("Initial Prior Probability for state 0 (not-HSP)",fontdict={"size":18})
plt.ylabel("Mean of the classification (action)\n by agents",fontdict={"size":18})
plt.yticks(Xs,labels_ticks,fontsize=12)
plt.xticks(fontsize=12)
plt.legend(title="Underlying State: Description",title_fontsize="large",fontsize="medium")
plt.savefig("plots/public_belief_grid.png")
