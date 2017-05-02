import gym
import numpy
from scipy import stats
#from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
import numpy as np
import random


env = gym.make('Pendulum-v0')
#env.monitor.start('/tmp/Cart_ml-submit-v1.02')

#classifier = RandomForestRegressor()  # Use a random forest
#classifier = linear_model.Lasso()
classifier = linear_model.SGDRegressor()
random.seed()

X = []  # active Independent / Input variables
Y = []  # active Dependent / Output variables
R = []  # active Rewards / Weight variables
R1 = [] # active Rewards / Weight variables

hist_observation = []
hist_actions = []
hist_rewards = [-100]
Old_observation = 0.0
reward = 0


RangeMax = 100000
Trained = 0
AVG_reward = [0] * 100
AVG_reward_count = 0

Solved_at = -1
Failed_at = -5
Solved_status = "Failed"   # Failed, Solved, Training

Old_action = env.action_space.sample()
action = env.action_space.sample()

for i_episode in xrange(1):
    observation = env.reset()
    X_tmp = []
    Y_tmp = []
    for t in xrange(RangeMax):
        env.render()

        (observation, reward, done, info) = env.step(action)

        hist_observation.append(observation)
        hist_actions.append(action)
        hist_rewards.append(reward)

        #print "reward :" +str(reward)

        if done:
                break

        #print " O:" + str(observation)
        # Calculate average of past 100 actions
        if len(AVG_reward) < 100:
            AVG_reward.append(t+1)
        else:
            AVG_reward[AVG_reward_count] = reward
            AVG_reward_count += 1
        if AVG_reward_count == 100:
            AVG_reward_count = 0
        Curr_AVG_reward = sum(AVG_reward) / len(AVG_reward)
        if Curr_AVG_reward > Solved_at:
            Solved_status = "Solved"
        elif Curr_AVG_reward < Failed_at:
            Solved_status = "Failed"
        else:
            Solved_status = "Training"

        if (t % 100 == 0):
            print ""
            print "Episode " + str(i_episode) + " after " + str(t + 1) + " timesteps.  Avg:" + \
                  str(Curr_AVG_reward) + "  len(Y):" + str(len(Y))

            # Prune some old entries to "try always new things"
            if len(Y) > 100:
                NumToDelete = int(len(Y) * .01 )
                for c in range(NumToDelete):
                    IndexToDelete = R.index(min(R))
                    del X[IndexToDelete]
                    del Y[IndexToDelete]
                    del R[IndexToDelete]
                    del R1[IndexToDelete]
                print "     +--> Delete " + str(NumToDelete) + " Entries"

            if len(Y) > 5000:
                NumToDelete = len(Y) - 4999
                for c in range(NumToDelete):
                    IndexToDelete = R.index(min(R))
                    if R1[IndexToDelete] == 0:
                        del X[IndexToDelete]
                        del Y[IndexToDelete]
                        del R[IndexToDelete]
                        del R1[IndexToDelete]


            if len(Y) > 10:
                classifier.fit(X, Y,sample_weight=R1)  # Generate Model
                #classifier.fit(X, Y)      # Generate Model
                Trained += 1
                #print "     +-- AVG reward :" + str(sum(AVG_reward) / len(AVG_reward))



        sum_observation = observation[2] # sum(observation)
        Old_observation = hist_observation[-1][2] # sum(hist_observation[-1])

        #if abs(sum_observation) < abs(Old_observation) and ((sum_observation <0 and Old_observation <0) or (sum_observation  >0 and Old_observation >0)):

        if 0 == 0:
        #if reward > hist_rewards[-2] and reward >-1:
            #print "Old reward :" +str(hist_rewards[-2]) + " reward :" +str(reward)
            #XX=raw_input("wait:")
            if len(hist_observation) > 7:
                if hist_rewards[-1] > hist_rewards[-2] and hist_rewards[-2] > hist_rewards[-3]:
                        #and hist_rewards[-3] > hist_rewards[-4] and hist_rewards[-4] > hist_rewards[-5]\
                        #and hist_rewards[-1] > Curr_AVG_reward:
                    X_tmp = []
                    for C1 in range(2,3):
                        for C2 in range(3):
                            X_tmp.append(hist_observation[-C1][C2])
                        X_tmp.append(float(hist_actions[-C1]))
                        #X_tmp.append(hist_rewards[-C1])

                    X.append(X_tmp)
                    Y.append(hist_actions[-1][0])
                    R.append(hist_rewards[-1] - hist_rewards[-2])
                    if hist_rewards[-1] > -1:
                        R1.append(1)
                    else:
                        R1.append(0)




        if Trained > 7 and (Solved_status == "Solved" or Solved_status == "Training"):
            # Use our prediction model
            X_tmp = []
            for C1 in range(1, 2):
                for C2 in range(3):
                    X_tmp.append(hist_observation[-C1][C2])
                X_tmp.append(float(hist_actions[-C1]))
                #X_tmp.append(hist_rewards[-C1])
            action[0] = classifier.predict(X_tmp)


            #if hist_rewards[-1] < hist_rewards[-2] and hist_rewards[-2] < hist_rewards[-3]\
            #        and hist_rewards[-3] < hist_rewards[-4] and hist_rewards[-4] < hist_rewards[-5]\
            #        and hist_rewards[-5] < hist_rewards[-6] and hist_rewards[-6] < hist_rewards[-7]:
            #    action = env.action_space.sample()
            #    print ".",

        else:
            action = env.action_space.sample()
            pass

    if t == RangeMax-1:
        Solved += 1
        print "     +--> Finished " +str(Solved)+" times."


#env.monitor.close()
