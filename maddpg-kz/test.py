import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
def test(env,ob,ex,acdim):

    actor1  = load_model('results/actor05000.h5')
    actor2  = load_model('results/actor15000.h5')
    actor3  = load_model('results/actor25000.h5')
    ac = [actor1,actor2,actor3]
    for i in range(50):
        s = env.reset()
        plt.close()
        plt.figure()
        for i in range(300):
            env.render(s)
            plt.clf()
            a = []
            for i in range(3):
                state = np.reshape(s[i],(-1, ob[i]))
                a.append((ac[i].predict(state) + ex[i]()).reshape(acdim[i],))
            s2, r, done = env.step(a)
            s = s2
            if sum(done):
                break