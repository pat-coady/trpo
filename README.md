# Summary

The **same** learning algorithm was used to train agents for each of the ten OpenAI Gym MuJoCo continuous control environments. The only difference between evaluations was the number of episodes used for training. The code is available in the [GitHub repository](https://github.com/pat-coady/trpo/tree/aigym_evaluation). The exact code used to generate the submissions is in the **`aigym_evaluation`** branch.

The README.md file in the GitHub repository provides additional details on the algorithm and usage instructions. Also, the code was written to be understandable and easily modifiable.

Here are the key points:

* Proximal Policy Optimization (similar to TRPO, but uses gradient descent with KL loss term)  \[1\] \[2\]
* Value function approximated with 3 hidden-layer NN (tanh activations):
    * hid1 size = obs_dim x 10
    * hid2 size = geometric mean of hid1 and hid3 sizes
    * hid3 size = 5
* Policy is a multi-variate Gaussian parameterized by a 3 hidden-layer NN (tanh activations):
    * hid1 size = obs_dim x 10
    * hid2 size = geometric mean of hid1 and hid3 sizes
    * hid3 size = action_dim x 10
    * Diagonal covariance matrix variables are separate from NN
* Generalized Advantage Estimation (gamma = 0.995, lambda = 0.98) \[3\] \[4\]
* ADAM optimizer used for both neural networks
* Policy is evaluated for 20 episodes, and then updated
* Value function is trained on current batch + previous batch
* Policy and Value NNs built on TensorFlow framework

**Below you will find various training curves for this environment.**

### References

1. [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) (Schulman et al., 2016)
2. [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf) (Heess et al., 2017)
3. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf)
4. [GitHub Repository with several helpful implementation ideas](https://github.com/joschu/modular_rl) (Schulman)