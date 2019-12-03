## Trust Region Policy Optimization with Generalized Advantage Estimation

By Patrick Coady: [Learning Artificial Intelligence](https://learningai.io/)

### Summary

**NOTE:** The code has been refactored to use TensorFlow 2.0 and PyBullet (instead of MuJoCo). See the `tf1_mujoco` branch for old version.

The project's original goal was to use the same algorithm to "solve" [10 MuJoCo robotic control environments](https://gym.openai.com/envs/#mujoco). And, specifically, to achieve this without hand-tuning the hyperparameters (network sizes, learning rates, and TRPO settings) for each environment. This is challenging because the environments range from a simple cart pole problem with a single control input to a humanoid with 17 controlled joints and 44 observed variables. The project was successful, nabbing top spots on almost all of the AI Gym MuJoCo leaderboards.

With the release of TensorFlow 2.0, I decided to dust off this project and upgrade the code. And, while I was at it, I moved from the paid MuJoCo simulator to the free PyBullet simulator.

Here are the key points:

* Trust Region Policy Optimization \[1\] \[2\]
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

### PyBullet Gym Environments

```
HumanoidDeepMimicBulletEnv-v1
CartPoleBulletEnv-v1
MinitaurBulletEnv-v0
MinitaurBulletDuckEnv-v0
RacecarBulletEnv-v0
RacecarZedBulletEnv-v0
KukaBulletEnv-v0
KukaCamBulletEnv-v0
InvertedPendulumBulletEnv-v0
InvertedDoublePendulumBulletEnv-v0
InvertedPendulumSwingupBulletEnv-v0
ReacherBulletEnv-v0
PusherBulletEnv-v0
ThrowerBulletEnv-v0
StrikerBulletEnv-v0
Walker2DBulletEnv-v0
HalfCheetahBulletEnv-v0
AntBulletEnv-v0
HopperBulletEnv-v0
HumanoidBulletEnv-v0
HumanoidFlagrunBulletEnv-v0
HumanoidFlagrunHarderBulletEnv-v0
```

### Using

I ran quick checks on three of the above environments and successfully stabilized a double-inverted pendulum and taught the "half cheetah" to run.

```
python train.py InvertedPendulumBulletEnv-v0
python train.py InvertedDoublePendulumBulletEnv-v0 -n 5000
python train.py HalfCheetahBulletEnv-v0 -n 5000 -b 5
```

### Videos

During training, videos are periodically saved automatically to the /tmp folder. These can be enjoyable to view, and also instructive.

### Dependencies

* Python 3.6
* The Usual Suspects: numpy, matplotlib, scipy
* TensorFlow 2.x
* Open AI Gym: [installation instructions](https://gym.openai.com/docs)
* [pybullet](https://pypi.org/project/pybullet/) physics simulator

### References

1. [Trust Region Policy Optimization](https://arxiv.org/pdf/1502.05477.pdf) (Schulman et al., 2016)
2. [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/pdf/1707.02286.pdf) (Heess et al., 2017)
3. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/pdf/1506.02438.pdf) (Schulman et al., 2016)
4. [GitHub Repository with several helpful implementation ideas](https://github.com/joschu/modular_rl) (Schulman)
