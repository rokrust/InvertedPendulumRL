import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
# import gym
# env = gym.make('CartPole-v0')
# env.reset()
# print(env.action_space)
# for _ in range(1000):
#     env.render()
#     observation = env.step(env.action_space.sample()) # take a random action
#     print(observation)
            
