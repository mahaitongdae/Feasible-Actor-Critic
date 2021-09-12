## Requirements

**Important information for installing the requirements:**
1. We test it successfully only on **Python 3.6**, and higher python version causes error with Safety Gym and TensorFlow 2.x.
2. Safety Gym and TensorFlow 2.x have conflict in numpy version. We test on numpy 1.17.5. If it runs with errors, pls check the numpy version.
To install requirements:

```setup
pip install -r requirements.txt
```
If error occurs when installing [MuJoCo and mujoco-py](https://github.com/openai/mujoco-py) in [Safety Gym](https://github.com/openai/safety-gym) dependencies, Please remove the mujoco-py from requirements.txt in Safety Gym and this repo. Then manually compile it from source. 

Replace the engine.py in 

```angular2html
$YOUR_SAFETY_GYM_LIBRARY_ROOT$/envs/engine.py
```
with our modified version, utils/engine.py

## Training

To train the model(s) in the paper, run this command:

```train
python train_scripts4fsac.py
```
Change the alg_name in main.py if training with different algorithm.
## Evaluation

To test and evaluate trained policies, change the mode in train_scripts4fsac.py to testing and run.