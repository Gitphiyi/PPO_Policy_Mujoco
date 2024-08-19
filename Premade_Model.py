import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
import mujoco
from stable_baselines3 import SAC, TD3, A2C
import os 
import argparse

model_dir = "models"
log_dir = "logs"
vids_dir = "vids"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(vids_dir, exist_ok=True)

def train(env):
    model = SAC("MlpPolicy", env, verbose=1, device='cuda', tensorboard_log=log_dir)

    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1
        print('hi')
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/SAC_{TIMESTEPS*iters}")

def test(env, path_to_model):
    model = SAC.load(path_to_model, env=env)
    obs = env.reset()[0]
    done = False
    extra_steps = 500 #tosee the human fall all the way to the ground bc it ends right as it falls
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done: 
            extra_steps -= 1
            if extra_steps < 0:
                break
    print("finished!") 

def create_video(env, path_to_model):
    model = SAC.load(path_to_model, env=env)
    obs = env.reset()[0]
    video = VideoRecorder(env, path=f'{vids_dir}/SAC_moving_bot.mp4')
    done = False
    extra_steps = 200 #tosee the human fall all the way to the ground bc it ends right as it falls
    while extra_steps >= 0:
        action, _ = model.predict(obs)
        #action = env.action_space.sample()
        obs, _, done, _, _ = env.step(action)
        video.capture_frame()

        if done: 
            extra_steps -= 1
    video.close()
    print("finished!") 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    parser.add_argument('-d', '--video', metavar='path_to_model')

    args = parser.parse_args()
    print("Gym version:", gym.__version__)
    print("MuJoCo-py version:", mujoco.__version__)
    if args.train:
        gymenv = gym.make(args.gymenv, render_mode = 'human') #this is a known error in https://github.com/Farama-Foundation/Gymnasium/issues/749https://github.com/Farama-Foundation/Gymnasium/issues/749
        train(gymenv)
    
    if args.test:
        if os.path.isfile(args.test):
            gymenv = gym.make(args.gymenv, render_mode='human')
            test(gymenv, path_to_model=args.test)

        else: 
            print(f'{args.test} not found.')

    if args.video:
        if os.path.isfile(args.video):
            gymenv = gym.make(args.gymenv, render_mode='rgb_array')
            create_video(env=gymenv, path_to_model=args.video)
        else: 
            print(f'{args.test} not found.')