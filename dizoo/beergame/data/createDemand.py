import numpy as np
import sys


# it gets an integer as input, which 0 means generate random numbers from a uniform distribution,
# and 1 means generate random numbers from normal distribution, 2 means the Sterman dataset 

distribution_type=int(sys.argv[1])


np.random.seed(seed=10)


sizee = 6100
# sizee = 60100
ifmulti_agent_train = False
mu = 10
sigma = 2 
demandUp = 3
ifStermanData = True
T=10002
# T=102

if distribution_type == 0:
	demandTr = np.random.randint(0,demandUp,size=[sizee,T])
	if ifmulti_agent_train:
		np.save('demandTr'+str(distribution_type)+'-'+str(demandUp)+'-'+str(sizee),demandTr)
	else:	
		np.save('demandTr'+str(distribution_type)+'-'+str(demandUp),demandTr)
elif distribution_type == 1:
	demandTr = np.round(np.random.normal(mu,sigma,size=[sizee,T])).astype(int)
	if ifmulti_agent_train:
		np.save('demandTr'+str(distribution_type)+'-'+str(mu)+'-'+str(sigma)+'-'+str(sizee),demandTr)
	else:	
		np.save('demandTr'+str(distribution_type)+'-'+str(mu)+'-'+str(sigma),demandTr)
elif distribution_type == 2:
	demandTr = np.concatenate((4*np.ones((sizee,4)) ,8*np.ones((sizee,98))), axis=1).astype(int)
	if ifmulti_agent_train:
		np.save('demandTr'+str(distribution_type)+'-'+str(sizee),demandTr)
	else:	
		np.save('demandTr'+str(distribution_type),demandTr)

if distribution_type == 0:
	demandTs = np.random.randint(0,demandUp,size=[1000,T])
	np.save('demandTs'+str(distribution_type)+'-'+str(demandUp),demandTs)
elif distribution_type == 1:
	demandTs = np.round(np.random.normal(mu,sigma,size=[1000,T])).astype(int)
	np.save('demandTs'+str(distribution_type)+'-'+str(mu)+'-'+str(sigma),demandTs)
elif distribution_type == 2:
	demandTs = np.concatenate((4*np.ones((1000,4)) ,8*np.ones((1000,98))), axis=1).astype(int)
	np.save('demandTs'+str(distribution_type),demandTs)