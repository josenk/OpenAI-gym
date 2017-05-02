import gym
import numpy
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import random


env = gym.make('CartPole-v0')
env.monitor.start('/tmp/Cart_ml-submit-v1.02')

classifier = RandomForestClassifier()  # Use a random forest classifier
random.seed()

X = []  # active Independent, or input variables
Y = []  # active Dependent, or output variables

RangeMax = 200
Trained = 0
Solved = 0
Episodes = [0] * 100
Episode_count = 0

Old_action = 0
action = 0

for i_episode in xrange(500):
    observation = env.reset()
    X_tmp = []
    Y_tmp = []
    for t in xrange(RangeMax):
        env.render()

        Old_observation = observation[2] + observation[3]

        (observation, reward, done, info) = env.step(action)

        if done or t == RangeMax -1:

            # Calculate average of past 100 actions
            if len(Episodes) < 100:
                Episodes.append(t+1)
            else:
                Episodes[Episode_count] = t+1
            Episode_count += 1
            if Episode_count == 100:
                Episode_count = 0

            print "Episode " + str(i_episode) + " after " + str(t + 1) + " timesteps.  Avg:" + \
                  str(sum(Episodes) / len(Episodes)) + "  len(Y):" + str(len(Y))

            # Prune some old entries to "try always new things"
            if len(Y) > 1000:
                NumToDelete=50
                if len(Y_tmp) > 100:
                     NumToDelete = int(len(Y_tmp) / 2)
                X = X[NumToDelete:]
                Y = Y[NumToDelete:]
                #print "     +--> Delete " + str(NumToDelete) + " Entries"

                if len(Y) > 10000:
                    X = X[-10000:]
                    Y = Y[-10000:]

            classifier.fit(X, Y)  # Generate Model
            Trained += 1
            if done:
                break


        sum_observation = observation[2] + observation[3]
        #  previous action gave good/bad results.   Is this breaking the rules???
        if abs(sum_observation) < abs(Old_observation):
            X.append(list(observation))
            Y.append(int(action))
        else:
            X.append(list(observation))
            Y.append(int(abs(action - 1)))

        if Trained >= 5:
            # Use our prediction model
            action = int(classifier.predict(list(observation)))
        else:
            # No prediction model is setup yet, so just go random
            action = random.randint(0,1)

    if t == RangeMax-1:
        Solved += 1
        print "     +--> Solved " +str(Solved)+" times."


env.monitor.close()
