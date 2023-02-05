## Neural networks

### Biological view
The basic computational unit of the brain is a __neuron__. Approx 86B in the human nervous system, connected with approx 
10^15 __synapses__.  
Each neuron receives input signals from its __dendrites__ and produces output signals along its __axon__. The axon 
connects to dendrites of other neurons via synapses.  

If we model the neuron, we can say that the signal that travel along the axon (__x0__) interact with the dendrites of 
other neuron (__w0 ⋅ x0__) based on the __synaptic strength__ of the synapse (__w0__).  
The idea is that the synaptic strengths (weight __w__) are learnable and control the strength of influence 
(and direction: excitory being positive weight, inhibitory being negative weight) of one neuron to another.  
When the signal is carried to the __cell body__ by the dendrite, the cell body add-up all interaction (wi ⋅ xi). If the 
sum is above a certain threshold, the neuron can __fire__, sending a spike along its axon. The __firing rate__ of the 
neuron is modeled by an activation function __f__.  

![neuron](neuron.png "") 

## Feed-Forward Neural Network

- n inputs x
- 3 neurons in a single hidden layer h
- 2 outputs y
- W1 as set of weights from x to h
- W2 as set of weights from h to y
  
![NN](NN.png "")  

Since there is only one hidden layer, there will be only 2 steps in the feedforward cycle:
- __Step 1__: Finding `h_bar` from input `x_bar` and set of weights `W1`
- __Step 2__: Finding the output `y_bar` from the calculated `h_bar` and the set of weights `W2`
  
### __Step 1__: Finding `h_bar`  
When we have more than one neuron in the hidden layer (3 here), h_bar is a vector.

We denote `W_ij` the weight that connect input `i` to the hidden neuron `j`. So the weight that connect input `2` to the 
hidden neuron `3` is denoted as `W_23`.  
  
The input vector `x_bar = [x1, x2, x3, ..., xn]`  
is multiplied by the weight matrix `W1 = [W_11, W_12, W_13; W_21, W_22, W_23; W_31, W_32, W_33; ...; W_n1, W_n2, W_n3 ]`  
to produce `h_bar' = [h'1, h'2, h'3]`  

<h3><center>h_bar' = x_bar ⋅ W1 </center></h3>  

To make sure the values of `h_bar` do not explode or increase too much in size, we use an __activation function__ `Φ`.  

<h3><center>h_bar = Φ(h_bar') </center></h3>   

### Example of __activation functions__
### The __hyperbolic tangent__: to ensure the output is between 1 and -1  

![Tanh](tanh.png "")   

***  
### The __sigmoid__: to ensure the output is between 1 and 0  

![Sigmoid](sigmoid.png "")   
#### Disadvantage
Let's denote the sigmoid function `f(x) = 1 / 1+e^-x`. If we derive the function we obtain `f'(x) = f(x)(1-f(x))`.  
The sigmoid here forces the model to "lose" information from the data. 
If we plot the derivative and think about the possible max value of the derivative of the sigmoid, the output is 
squeezed by at least one quarter at each layer during __backpropagation__, this can become a huge loss of information in
deeper neural network.   

Sigmoid being between 0 and 1, we can see that the max value of the derivative is 0.25.
![Derivative sigmoid](derivative_sigmoid.png "")  
So we avoid using the Sigmoid in DNN as activation functions for hidden units.  

***  
### The __Rectified Linear unit (ReLu)__: to ensure negative values to be 0 and positive values remain the same  

![ReLu](relu.png "")   
#### Advantage
Faster during training, good for deep neural network since the max of the derivative is 1, so no squeezing effect of the error 
during __backpropagation__.
#### Disadvantage  
If the learning rate is too high, ReLu units becomes fragile during the training phase and can die.
large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never 
activate on any datapoint again. If this happens, then the gradient flowing through the unit will forever be zero from 
that point on (by Andrej Karpathy [here](https://cs231n.github.io/neural-networks-1/#nn))
  


### In short
They all allow the network to represent nonlinear relationships between its inputs and outputs (crucial because most 
real word data is nonlinear).
But using them is tricky since they contribute to the __vanishing gradient problem__.  
#### Which one should I pick?
TLDR: Use the ReLU non-linearity, be careful with your learning rates and possibly monitor the fraction of “dead” units 
in a network. If this concerns you, give Leaky ReLU or Maxout a try. Never use sigmoid. Try tanh, but expect it to work 
worse than ReLU/Maxout.  

### __Step 2__: Finding `y_bar`  

Mathematically the idea is the same as for __finding h_bar in step 1__  
The input vector `h_bar = [h1 h2 h3]`  
is multiplied by the weight matrix `W2 = [W_11 W_12 W_13; W_21 W_22 W_23; W_31 W_32 W_33;]`  
to produce `y_bar' = [y1, y2]`  

<h3><center>y_bar = h_bar ⋅ W2 </center></h3> 

Once `y_bar` found, adding an __activation function__ is optional. In some problems, we can use the __softmax function__
(ie. multiclass classification).  
The softmax will allow the values to be between 0 and 1 and the sum of the values will be 1 (good for probabilities).

![Softmax](softmax.png "")   

## Backpropagation  

- We have computed the output via the feed-forward pass
- We have computed the error __E__ (ie difference between ground truth and the output of the network)

Now we have to go __backward__ in order to change the __weights__ in the goal of decreasing the network error __E__.  
Going backward from the outputs to the inputs while changing the weights is a process called __backpropagation__ 
(which is __Stochastic Gradient Descent computed using the chain rule__).  

__The goal is to find a set of weight that minimizing the network error__  
We use an iterative process presenting the network with one input at a time from the training set.  

Let's consider the error __EA__ (at point A) parameterized by the weight __WA__ after a forward-pass (The error that we 
have when having the weight WA)  

![backprop1](backprop1.png "")  

To reduce the error, we need to increase the weights.  
Since the gradient __∇__ (the derivative, or slope of the curve) at point A is __negative__ (pointing down), we need to change 
the weight in its __negative direction__ to increase the value of WA.  

![backprop2](backprop2.png "")  
  
At point B, the gradient is positive, so if we update the weight in the negative direction of the gradient, it will 
decrease the weight __WB__, thus the error __EB__.  

![backprop3](backprop3.png "")  

The weight update of a single weight is represented as follow:  
<h3><center> W_new = W_prev + α(-∂E/∂W) </center></h3>   

- __α__: learning rate  
- __∂E/∂W__: partial derivative of the error with respect to the weight (how each weight __separately__ change the error 
since the error is a function of many variables)  
- __α(-∂E/∂W) = ΔW_ijk__: backpropagation (amount by which the weight will be updated between layer k at neuron i and 
layer k+1 at neuron j)  
  
  


























