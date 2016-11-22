from rateSimulator.rateSimulator import *
import matplotlib.pyplot as plt

np.random.seed(100)             # set random number seed for reproducibility of simulations

# Simulation parameters

simtime = 1.0                   # s # Simulation time
dt = 1e-3                       # default sim time step

# Network parameters: numbers

Ninp = 10                       # Total number of input neurons
N = 20                          # Total number of neurons in next layer

# Rate neuron parameters

taum = 20e-3                    # s # Membrane time constant
bias = 0.                       # constant bias input to all neurons
maxR = 250.0                    # Hz # max firing rate
linRange = maxR*5               # Hz # width of the linear range of rate sigmoid
                                # outputrate = maxR*tanh(inp/linRange+bias)

# make an umbrella model with neuron groups, synapses and probes
model = Model(dt)

# Initialize neuron groups

# input layer
PinpA = ConstantRateNeuronGroup(Ninp,rates=10)
PinpA.r = np.ones(Ninp)*10      # 1Hz constant input
# next layer
PA = RateNeuronGroup(N,taum,maxR,bias,linRange,model,'PA')
 
# constant synapses
con_PinpA_PA = Synapses(PinpA,PA,model,'inpA_A')

## plastic synapses
#con_PinpA_PA = SynapsesPlasticBCM(PinpA,PA,model,'inpA_A',
#                    eta,kernelPotnIntegral,kernelDepnIntegral,\
#                    slowPostKernelIntegral,wmax,\
#                    wsyn0,purepostthresh,pureposteta,pureprethresh)

winit = 1.0                     # synaptic weight
# connect synapses - each post-synaptic neuron will receive Ninp connections
con_PinpA_PA.connect(Ninp,winit)  # all to all

# Input probe
PA_inp = InpProbe(PA,model,'PA_inp')

# Rate probe
PA_poprate = PopRateProbe(PA,model,'PA_poprate')

# mean weight probe
PinpA_PA_wsyn = SynapsesMeanProbe(con_PinpA_PA,model,'con_inpA_A')

# simulate
model.simulate(simtime)

# plot
trange = np.linspace(0,simtime,len(PA_poprate.data))

fig = plt.figure()
plt.subplot(121)
plt.plot(trange,PA_poprate.data,color='r',label="PA rate")
plt.ylabel("Hz")
plt.xlabel("Time (s)")
plt.legend()
plt.subplot(122)
plt.plot(trange,PinpA_PA_wsyn.data,color='r',label="PinpA to PA") # inpA to PA mean input weight
plt.ylabel("Weight (arb)")
plt.xlabel("Time (s)")
plt.legend()
fig.tight_layout()

plt.show()
