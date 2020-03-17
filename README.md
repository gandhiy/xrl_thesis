


### Tested




### Testing
- shap values for breakout and other atari games


### Need to work on




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

* using nightly stable-baselines (2.10.0a0) to be able to implement custom callback with CallBack List

* auto set saved path to saved_model folder. To change, set `model_path` in `~/models/base.py`. 

* go back to pybullet at somepoint
* project uses mujoco, atari, robotics, and other envs


### TODO: 
- [x] get all keys and sub-keys
- [x] get objects (even in sub dictionaries)
- [x] process array inputs
- [x] build permutations from all arrays
- [x] build all possible dictionaries
- [x] add shap value callback 
- [x] allow config to set the value of the number of shap training and testing samples
- [x] custom DQN Network
- [x] custom DDPG Network
- [ ] hyper parameter tune a model on a environment
- [ ] hyper parameter tune a model on multiple (acceptable) environments
- [ ] add shap value parameter
- [ ] add shap value display
- [ ] Humanoid dodge ball
- [ ] add toribash gym environments
- [ ] atlas gym environment








