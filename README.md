


### Tested
- A2C with atari environment and a step parameter on timesteps
- PPO1 with mujoco humanoid environment, but no multi dictionary parameter and no render
- PPO1 with mujoco humanoid environment with render testing, but no multi dictionary parameter
- PPO1 does not have a replay buffer.
- shap values without using replay buffer
- main algo works on brute! 



### Testing
- shap values for breakout and other atari games


### Need to work on
- tb logging and also displaying an intermediate rendering step
- shap values with/without weighted backgrounds



 ### Notes:
 * To install openai/mujoco envs with mujoco200, I had to go inside of the gym repo setup.py file and change the requirements for mujoco-py & robotics to be < 2.1 not < 2.0. Hopefully this does not cause any future errors?

Initially
```yaml
'mujoco': ['mujoco_py>=1.50, <2.0', 'imageio'],
  'robotics': ['mujoco_py>=1.50, <2.0', 'imageio'],
```

The install for mujoco gym envs and robotics gym envs with mujoco_py 2.0 and mujoco200 worked when I changed the above to
```yaml
'mujoco': ['mujoco_py>=1.50, <2.1', 'imageio'],
  'robotics': ['mujoco_py>=1.50, <2.1', 'imageio'],
```

Run inside of gym/
```bash
pip install -e .[mujoco] && pip install -e .[robotics]
```

* go back to pybullet at somepoint
* project uses mujoco, atari, robotics, and other envs


### TODO: 
- [x] get all keys and sub-keys
- [x] get objects (even in sub dictionaries)
- [x] process array inputs
- [x] build permutations from all arrays
- [x] build all possible dictionaries
- [x] add shap value callback 
- [ ] allow config to set the value of the number of shap training and testing samples
- [ ] hyper parameter tune a model on a environment
- [ ] hyper parameter tune a model on multiple (acceptable) environments
- [ ] add shap value parameter
- [ ] add shap value display
- [ ] Humanoid dodge ball
- [ ] add toribash gym environments
- [ ] atlas gym environment








