from time import time
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Proportion of test data of all the data
PROPORTION_TEST = 0.3

train_in, train_out, test_in, test_out, predicted = [], [], [], [], []
file = open('data.txt', 'r')
for line in file:
    nums = list(map(lambda x: int(x), line.split()))
    coin = np.random.rand()
    if coin > PROPORTION_TEST:
        train_in.append(nums[:-1])
        train_out.append(nums[-1:])
    else:
        test_in.append(nums[:-1])
        test_out.append(nums[-1:][0])


def train_universal_model():
    print('Started training universal model')
    universal_model = MLPRegressor(hidden_layer_sizes=(3,),
                                   activation='relu',
                                   solver='adam',
                                   learning_rate='adaptive',
                                   max_iter=1000,
                                   learning_rate_init=0.01,
                                   alpha=0.01)
    start_time = int(time() * 1000)
    universal_model.fit(train_in, train_out)
    end_time = int(time() * 1000)
    print('Finished training universal model')
    print('Training took {} ms'.format(end_time - start_time))
    return universal_model.predict(test_in)
    # self.set_universal_model_thread_safe(universal_model)


predicted = train_universal_model()
print('mean_squared_error: ', mean_squared_error(test_out, predicted))
