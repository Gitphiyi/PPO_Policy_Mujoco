import gymnasium as gym
import ppo
from pathlib import Path
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder



model_dir = "models"
vids_dir = "vids"

if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    model = ppo.PPO(env)
    timesteps =  15000000
    load_steps = 15000000
    pth = Path('models') / f'ppo_{load_steps}_lr=0.001.pth'
    model.learn(timesteps)
    model.save_models(timesteps)
    #model.load_model(pth)    
    print("model is trained")

    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    video = VideoRecorder(env, path=f'{vids_dir}/pendulum.mp4')

    obs = env.reset()[0]
    max_steps = 2000
    steps  = 0
    while steps < max_steps:
        action, _ = model.get_action(obs)
        obs, rew, done, _, _ = env.step(action)
        print(rew)
        video.capture_frame()
        steps += 1
    video.close()
    print("finished!") 
