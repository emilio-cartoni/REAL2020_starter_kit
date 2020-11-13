from baseline.abstractor import VAEAbstractor
from baseline.abstractor import currentAbstraction
import numpy as np
import baseline.config as config
import baseline.priorityQueue as pq
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    allActions = np.load(config.sim['experience_data'], allow_pickle=True).tolist()

    allAbstractedActions = [[currentAbstraction(a[0]), a[1],
                             currentAbstraction(a[2])]
                            for a in allActions]

    images = [action[2] for action in allAbstractedActions]
    print("Total images {} for VAE and BS".format(len(images)))
    cbsm = cv2.createBackgroundSubtractorMOG2(len(images))
    for i in np.random.permutation(len(images)):
        cbsm.apply(images[i])
    background = cbsm.getBackgroundImage()
    plt.imsave('bgvaetrain.png', background)
    images = [cbsm.apply(image, learningRate=0) / 255 for image in images]
    ab = VAEAbstractor(images, latent_dim=7 * config.abst['n_obj'], retrain=True)
