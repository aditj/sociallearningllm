import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
import seaborn as sns
import cvxpy as cp
import numpy as np


def cvxpy_nias(p_m_x_given_u, act_prob, act_sels, priors, margin_niac, margin_nias):
  """
  Formulates the NIAS constraints in CVXPY notation.

  Args:
    p_m_x_given_u: A 3D NumPy array representing the conditional probabilities
      p_m(x|u). The dimensions of the array should be (M, X, A), where M is the
      number of elements in set M, X is the number of elements in set X, and A
      is the number of elements in set U.
    margin_nias: The margin value for the NIAS constraint.

  Returns:
    A list of CVXPY constraints representing the NIAS inequalities.
  """

  # Extract dimensions
  M, X, A = p_m_x_given_u.shape
  r_m_u_x = cp.Variable((M*X*A+M, 1))

  # Create decision variable with shape (M*X*A, 1)
  

  # Constraints
  constraints = []

  # NIAS constraint
  for m in range(M):
    for u in range(A):
      for u_prime in range(A):
        if u != u_prime:  # Add condition to exclude u == u_prime
          nias_expr = 0
          for x in range(X):
            index = m*X*A + x*A + u
            index_prime = m*X*A + x*A + u_prime
            nias_expr += p_m_x_given_u[m, x, u] * (r_m_u_x[index_prime] - r_m_u_x[index])
          constraints.append(nias_expr + margin_nias <= 0)  # Add margin_nias here



  # NIAC constraint
  for m1 in range(M):
    for m2 in range(M):
      if m1 != m2:
        niac_expr = 0
        for u in range(A):
          max_term = -np.inf
          for u_prime in range(A):
            term = 0
            for x in range(X):
              index = m1*X*A + x*A + u_prime
              term += act_sels[m2][x, u] * priors[m2][x] / act_prob[m2][u] * r_m_u_x[index]
            max_term = cp.maximum(max_term, term) - r_m_u_x[M*X*A+m2]
          niac_expr += act_prob[m1][u] * max_term

          for x in range(X):
            index = m1*X*A + x*A + u
            niac_expr -= p_m_x_given_u[m1, x, u] * act_prob[m1][u] * r_m_u_x[index] 
          niac_expr -=  r_m_u_x[M*X*A+m1]
        constraints.append(niac_expr + margin_niac <= 0)  # Add margin_niac here
        constraints.append(r_m_u_x>=0)
        objective = cp.Minimize(cp.norm1(r_m_u_x[:M*X*A]))
        problem = cp.Problem(objective, constraints)
        problem.solve()
  return r_m_u_x.value


def cvxpy_nias_margin(p_m_x_given_u, act_prob, act_sels, priors, margin_niac, margin_nias):
  """
  Formulates the NIAS constraints in CVXPY notation.

  Args:
    p_m_x_given_u: A 3D NumPy array representing the conditional probabilities
      p_m(x|u). The dimensions of the array should be (M, X, A), where M is the
      number of elements in set M, X is the number of elements in set X, and A
      is the number of elements in set U.
    margin_nias: The margin value for the NIAS constraint.

  Returns:
    A list of CVXPY constraints representing the NIAS inequalities.
  """

  # Extract dimensions
  M, X, A = p_m_x_given_u.shape
  r_m_u_x = cp.Variable((M*X*A+M+2, 1))

  # Create decision variable with shape (M*X*A, 1)
  

  # Constraints
  constraints = []

  # NIAS constraint
  for m in range(M):
    for u in range(A):
      for u_prime in range(A):
        if u != u_prime:  # Add condition to exclude u == u_prime
          nias_expr = 0
          for x in range(X):
            index = m*X*A + x*A + u
            index_prime = m*X*A + x*A + u_prime
            nias_expr += p_m_x_given_u[m, x, u] * (r_m_u_x[index_prime] - r_m_u_x[index])
          constraints.append(nias_expr  + r_m_u_x[-1] <= 0)  # Add margin_nias here



  # NIAC constraint
  for m1 in range(M):
    for m2 in range(M):
      if m1 != m2:
        niac_expr = 0
        for u in range(A):
          max_term = -np.inf
          for u_prime in range(A):
            term = 0
            for x in range(X):
              index = m1*X*A + x*A + u_prime
              term += act_sels[m2][x, u] * priors[m2][x] / act_prob[m2][u] * r_m_u_x[index]
            max_term = cp.maximum(max_term, term) - r_m_u_x[M*X*A+m2]
          niac_expr += act_prob[m1][u] * max_term

          for x in range(X):
            index = m1*X*A + x*A + u
            niac_expr -= p_m_x_given_u[m1, x, u] * act_prob[m1][u] * r_m_u_x[index] 
          niac_expr -=  r_m_u_x[M*X*A+m1]
        constraints.append(niac_expr + r_m_u_x[-2] <= 0)  # Add margin_niac here
        constraints.append(r_m_u_x>=0)
        objective = cp.Maximize(r_m_u_x[-1]+r_m_u_x[-2])
        problem = cp.Problem(objective, constraints)
        problem.solve()
  return r_m_u_x.value

def return_utilities(p_m_x_given_u, margin_nias, margin_niac,M,X,A,priors = None):
    actsels_prob = p_m_x_given_u/p_m_x_given_u.sum(axis=2).reshape(M,A,1)
    actsels = actsels_prob
    if priors is None:
        priors = np.ones((M,X))/X

    act_probs_by_category = {}
    num_actions = actsels_prob.shape[-1]

    for categ in range(M): # parametrizes the prior
    act_probs_by_category[categ] = []
    for action in range(num_actions):
        # compute p(a) = prior*p(a|x)
        act_probs_by_category[categ].append(priors[categ,:]@actsels_prob[categ,:,action])
        # compute p(x|a) = p(a|x)*prior/p(a)

    act_probs_by_category[categ] = np.array(act_probs_by_category[categ])


    r = cvxpy_nias(p_m_x_given_u, act_probs_by_category, actsels, priors, margin_niac, margin_nias)
def return_utilities_max_margin(p_m_x_given_u, M,X,A,priors = None):
    actsels_prob = p_m_x_given_u/p_m_x_given_u.sum(axis=2).reshape(M,A,1)
    actsels = actsels_prob
    if priors is None:
        priors = np.ones((M,X))/X

    act_probs_by_category = {}
    num_actions = actsels_prob.shape[-1]

    for categ in range(M): # parametrizes the prior
    act_probs_by_category[categ] = []
    for action in range(num_actions):
        # compute p(a) = prior*p(a|x)
        act_probs_by_category[categ].append(priors[categ,:]@actsels_prob[categ,:,action])
        # compute p(x|a) = p(a|x)*prior/p(a)

    act_probs_by_category[categ] = np.array(act_probs_by_category[categ])


    r = cvxpy_nias(p_m_x_given_u, act_probs_by_category, actsels, priors)