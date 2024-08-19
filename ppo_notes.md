# Explaining Whole PPO Process

## Resources:
- https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-2-4-f9d8b8aa938a
- https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

# Algorithm 1 PPO-Clip
1. **Input:** initial policy parameters θ₀, initial value function parameters ϕ₀
2. **for** k = 0, 1, 2, ... **do**
   - **Collect** set of trajectories Dₖ = {τᵢ} by running policy πₖ = π(θₖ) in the environment.
   - **Compute** rewards-to-go R̂ₜ.
   - **Compute** advantage estimates, Âₜ (using any method of advantage estimation) based on the current value function Vϕₖ.
   - **Update** the policy by maximizing the PPO-Clip objective:
     ```
     θₖ₊₁ = arg max_θ ( 1 / |Dₖ|T ) ∑_(τ∈Dₖ) ∑_(t=0)^T min( (πθ(aₜ|sₜ) / πθₖ(aₜ|sₜ)) g(ε, Âπθₖ(sₜ, aₜ)), g(ε, Âπθₖ(sₜ, aₜ)) ),
     ```
     typically via stochastic gradient ascent with Adam.
   - **Fit** value function by regression on mean-squared error:
     ```
     ϕₖ₊₁ = arg min_ϕ ( 1 / |Dₖ|T ) ∑_(τ∈Dₖ) ∑_(t=0)^T ( Vϕ(sₜ) - R̂ₜ )²,
     ```
     typically via some gradient descent algorithm.
3. **end for**


1. Initializing both the actor and critic networks

2. For loop that trains model for an infinite amount of iterations. The number of iterations can also be set

3. Based on the current policy (πₖ = π(θₖ)) where θₖ is the policy parameters at the kth iteration, sample actions from the distribution created by the policy and record all the trajectories into a set or list. Trajectories are sometimes called episodes, so basically the robot records all its states and actions from beginning to end after doing a policy

4. This calculates rewards-to-go. Basically this means that for every state and action that the policy does there is a discount rate weighted sum of the rewards. This means that the actions the policy does earlier matter more because you want to have the policy do better actions earlier than later.

5. This calculates the advantage. The equation is basically A(s,a)=Q(s,a)−V(s). Q(s,a) is the expected return (total discounted rewards) after doing action a in state s and V(s) is the expected return of running the policy on state s. This is basically the expected return of actor model - value of critic model.

6. This step is essentially trying to maximize the policy at the timestep to create the best return using gradient ascent. Inside the min function the possible changes that can be made to the policy parameters are clipped. Basically the policy can either descend back to its previous state or it can improve by only a certain amount, which prevents too large of changes to the policy from happening. This helps the policy not leave an action space and basically stop learning because it did something too drastic.
Now let's understand the double summation. First, note that T is the trajectory length or the # of timesteps it took to end each episode. The inner summation will basically sum up the advantage*probability ratio, which I think basically if it is positive says that return after that action is good. I think a probability ratio is used because it prevents probabilities that were initialized high from continuing to be stepped into that direction. It might be explained in this link beter (https://towardsdatascience.com/an-intuitive-explanation-of-policy-gradient-part-1-reinforce-aa4392cbfd3c). 
The outer summation will sum up all the trajectories and the return from every trajectory. Then gradient ascent is run on the entire expression, which will optimize every policy parameter in each sum to maximize the return. This will create the new policy for the next time step. We then divide by the # of episode timesteps * # of trajectories to find a weighted average or Expected value of return.

7. This is training the critic network. Basically it is finding the expected mean square error amoung all the trajectories and then minimizing the critic parameters so the error is decreasing. To further simplify this, it is training the critic network to be more accurate at determining the amount of reward the policy should earn at a state.