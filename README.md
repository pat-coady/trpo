## Proximal Policy Optimization with Generalized Advantage Estimation

By Patrick Coady: [Learning Artificial Intelligence](https://learningai.io/)

### Summary

The same learning algorithm was used to train agents for each of the ten OpenAI Gym MuJoCo continuous control environments. The only difference between evaluations was the number of episodes used per training batch, otherwise all options were the same. The exact code used to generate the OpenAI Gym submissions is in the **`aigym_evaluation`** branch.

Here are the key points:

* Proximal Policy Optimization (similar to TRPO, but uses gradient descent with KL loss terms)  \[1\] \[2\]
* Value function approximated with 3 hidden-layer NN (tanh activations):
    * hid1 size = obs_dim x 10
    * hid2 size = geometric mean of hid1 and hid3 sizes
    * hid3 size = 5
* Policy is a multi-variate Gaussian parameterized by a 3 hidden-layer NN (tanh activations):
    * hid1 size = obs_dim x 10
    * hid2 size = geometric mean of hid1 and hid3 sizes
    * hid3 size = action_dim x 10
    * Diagonal covariance matrix variables are separately trained
* Generalized Advantage Estimation (gamma = 0.995, lambda = 0.98) \[3\] \[4\]
* ADAM optimizer used for both neural networks
* The policy is evaluated for 20 episodes between updates, except:
    * 50 episodes for Reacher
    * 5 episodes for Swimmer
    * 5 episodes for HalfCheetah
    * 5 episodes for HumanoidStandup
* Value function is trained on current batch + previous batch
* KL loss factor and ADAM learning rate are dynamically adjusted during training
* Policy and Value NNs built with TensorFlow

## Dependencies

* Python 3.5
* The Usual Suspects: NumPy, matplotlib, scipy
* TensorFlow
* gym - [installation instructions](https://gym.openai.com/docs)
* [MuJoCo](http://www.mujoco.org/) (30-day trial available and free to students)

### Results can be reproduced as follows:

```
./train.py Reacher-v1 -n 60000 -b 50
./train.py InvertedPendulum-v1
./train.py InvertedDoublePendulum-v1 -n 12000
./train.py Swimmer-v1 -n 2500 -b 5
./train.py Hopper-v1 -n 30000
./train.py HalfCheetah-v1 -n 3000 -b 5
./train.py Walker2d-v1 -n 25000
./train.py Ant-v1 -n 100000
./train.py Humanoid-v1 -n 200000
./train.py HumanoidStandup-v1 -n 200000 -b 5
```

### View the videos

During training, videos are periodically saved automatically to the /tmp folder. These can be enjoyable, and also instructive.

### References

1. [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) (Schulman et al., 2016)
2. [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf) (Heess et al., 2017)
3. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf) (Schulman et al., 2016)
4. [GitHub Repository with several helpful implementation ideas](https://github.com/joschu/modular_rl) (Schulman)
