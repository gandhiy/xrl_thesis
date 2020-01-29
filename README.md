List of environments



Tested



Notes:
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



TODO: 
- [x] get all keys and sub-keys
- [x] get objects (even in sub dictionaries)
- [x] process array inputs
- [x] build permutations from all arrays
- [x] build all possible dictionaries