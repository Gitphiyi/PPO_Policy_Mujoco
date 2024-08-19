import torch 
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from torch.optim.adam import Adam
from network import FNN
from pathlib import Path


class PPO:
    def __init__(self, env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.actor = FNN(self.obs_dim, self.act_dim) #in continuous action space usually dimensions are mean and std deviation to show best probability space
        self.critic = FNN(self.obs_dim, 1) #output layer is one dim for an unambiguous return. Used for advantage calculations
        self.__init_hyperparameters()

        self.cov_var = torch.full(size = (self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = self.lr)

    def __init_hyperparameters(self):
        self.max_timesteps_per_batch = 3000
        self.max_timesteps_per_episode = 1000
        self.gamma = 0.99 #this is discount rate for reward
        self.n_updates_per_iteration = 5
        self.clip = 0.3
        self.lr = 3e-4

    def save_models(self, timesteps):
        pth = Path('models') / f'ppo_{timesteps}_lr={self.lr}.pth'
        torch.save(self.actor.state_dict(), pth)
    
    def load_model(self, pth):
        self.actor.load_state_dict(torch.load(pth))

    def learn(self, total_timesteps):
        timestep_count = 0
        pth = Path('models') / f'ppo_{timestep_count}.pth'
        #self.load_model(pth)
        while timestep_count < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout() 
            timestep_count += np.sum(batch_lens)

            batch_value, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - batch_value.detach()
            A_k = (A_k - A_k.mean() ) / (A_k.std() + 1e-10) #normalize advantages

            for _ in range(self.n_updates_per_iteration):
                #no need for for loop because all calculations can be done in tensors
                batch_value, curr_log_prob = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_prob - batch_log_probs)

                default_loss = ratios * A_k 
                clipped_loss = torch.clamp(ratios, 1-self.clip, 1+self.clip) * A_k 
                actor_loss = (-torch.min(default_loss, clipped_loss)).mean() #the mean() is the same as dividing by (|Dk|T)
                critic_loss = nn.MSELoss()(batch_value, batch_rtgs)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            if timestep_count % 500000 == 0:
                print(f'actor loss: {actor_loss} \t critic loss: {critic_loss}')
            if timestep_count % 50000 == 0:
                print(f'trained until step {timestep_count}')
                self.save_models(timestep_count)
            
    def rollout(self):
        batch_obs = [] #batch observations
        batch_acts = [] #batch actions
        batch_log_probs = []
        batch_rews = [] #rewards
        batch_rtgs = [] #rewards to go
        batch_lens = [] #episodic lengths in batch

        obs = self.env.reset()
        done = False

        t = 0 #timesteps in a batch
        while t < self.max_timesteps_per_batch:
            ep_rew = []
            obs = self.env.reset()[0] #reset environment for every run episode. Should this be seeded?

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs)
                action, log_prob = self.get_action(obs)
                obs, rew, done, _, _ = self.env.step(action)

                ep_rew.append(rew)
                batch_acts.append(action)   
                batch_log_probs.append(log_prob)
                if done:
                    break
            batch_lens.append(ep_t+1)
            batch_rews.append(ep_rew)

        batch_obs = np.array(batch_obs)
        batch_acts = np.array(batch_acts) #converts list of np arrays to an np array of arrays, since apparently its faster than converting python list of np arrays to a tensor
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    
    def compute_rtgs(self, batch_rews):
        """
            Create a rewards-to-go for every timestep in an episode. Bit confused on why it is reversed and why not just append. Also why is the last reward in every episode weighted the most?
        """
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            discount_reward = 0
            for rews in reversed(ep_rews):
                discount_reward += self.gamma * rews
                batch_rtgs.insert(0, discount_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def get_action(self, obs):
        mean = self.actor.forward(obs) 
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_obs, batch_acts):
        mean = self.actor(batch_obs) 
        dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(batch_acts)
        V = self.critic(batch_obs).squeeze()
        return V, log_prob
