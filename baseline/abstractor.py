from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import numpy as np
import baseline.config as config
import baseline.priorityQueue as pq
import cv2

def currentAbstraction(obs):
    '''
    Extract the coordinates from the input (observation) taking only x, y, z depending on the config file

    Args:
        obs (list): list of observations in different form (image, coordinates, mask)

    Returns:
        coordinates list (list)          
    '''
    if config.abst['type'] == 'pos':
        return abstractionFromPos(obs[1])
    elif config.abst['type'] == 'pos_noq':
        return abstractionFromPosNoQ(obs[1])
    elif config.abst['type'] == 'mask':
        return obs[2] + 1
    elif config.abst['type'] == 'filtered_mask':
        return obs[2] == 2
    elif config.abst['type'] == 'image':
        return obs[0]

def abstractionFromPos(pos_dict):
    '''
    Extract the coordinates from the input 

    Args:
        pos_dict (dictionary): dictionary = {object0:coords0,...,objectN:coordsN}

    Returns:
        coordinates list (list): [coords0,...,coordsN]         
    '''
    abst = np.hstack([pos_dict[obj] for obj in ['cube','tomato','mustard'] if obj in pos_dict])
    abst = np.round(abst, config.abst['precision'])
    return abst

def abstractionFromPosNoQ(pos_dict):
    '''
    Extract the coordinates from the input taking only x, y, z 

    Args:
        pos_dict (dictionary): dictionary = {object0:coords0,...,objectN:coordsN}

    Returns:
        coordinates list (list): [coords0[:2],...,coordsN[:2]]         
    '''
    abst = np.hstack([pos_dict[obj][:2] for obj in ['cube','tomato','mustard'] if obj in pos_dict])
    abst = np.round(abst, config.abst['precision'])
    return abst




class Abstractor():
    def __init__(self, masks):
        # reparameterization trick
        # instead of sampling from Q(z|X), sample epsilon = N(0,I)
        # z = z_mean + sqrt(var) * epsilon
        def sampling(args):
            """Reparameterization trick by sampling from an isotropic unit Gaussian.

            # Arguments
                args (tensor): mean and log of variance of Q(z|X)

            # Returns
                z (tensor): sampled latent vector
            """

            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            # by default, random_normal has mean = 0 and std = 1.0
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon


        fl = masks

            

        x_train = fl[:int(np.floor(len(fl)*0.80))]
        x_test = fl[int(np.ceil(len(fl)*0.80)):]
        image_rows = x_train[0].shape[0] 
        image_columns = x_train[0].shape[1] 
        original_dim = image_rows * image_columns 
        #if config.abst['type'] == 'image':
            #original_dim *= 3
        x_train = np.reshape(x_train, [-1, original_dim]) 
        x_test = np.reshape(x_test, [-1, original_dim]) 
        
        if config.abst['type'] == 'mask':
            maximum = np.max([np.max(mask) for mask in fl])
            x_train = x_train.astype('float32') / maximum
            x_test = x_test.astype('float32') / maximum
        #elif config.abst['type'] == 'image':
        #    x_train = x_train.astype('float32') / 255.
        #    x_test = x_test.astype('float32') / 255.

        input_shape = (original_dim, ) 
        intermediate_dim = 512
        batch_size = 128 
        latent_dim = 6
        epochs = 30

        # VAE model = encoder + decoder
        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()
        #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()

        # instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        models = (self.encoder, self.decoder)

        # VAE loss = mse_loss or xent_loss + kl_loss
        if False:
            reconstruction_loss = mse(inputs, outputs)
        else:
            reconstruction_loss = binary_crossentropy(inputs,
                                                      outputs)

        reconstruction_loss *= original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        vae.summary()
        plot_model(vae,
                   to_file='vae_mlp.png',
                   show_shapes=True)


        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_abstraction_from_binary_image(self, image):
        return self.encoder.predict(np.reshape(image,[-1,len(image)*len(image[0])]))

    def get_abstraction_from_mask(self, mask):
        #here implement preprocessing 
        return self.encoder.predict(np.reshape(mask,[-1,len(mask)*len(mask[0])]))
    

class DynamicAbstractor():
    '''
    This class allows you to gradually release the description of the states contained in the actions.

    Args:
        actions (list): a list of (precondition, action, postcondition), where conditions are float lists 

    Attributes:
        actions (list): where the input actions are saved
        dictionary_abstract_actions (dict): where abstract actions are saved. The keys will be the levels of abstraction and the values ​​will be the lists of abstract actions
        lists_significative_differences (list): where significant distances are saved for each condition variable. These will be used to relax conditions.
    '''
    def __init__(self, actions):
        if len(actions[0]) != 3:
            print("Legal actions consist of (precondition,action,postcondition)")
            return None
        
        if not isinstance(actions[0][0], type(np.array([]))) or not isinstance(actions[0][2], type(np.array([]))):
            print("Each conditions have to be numpy.ndarray")
            return None         
        

        if config.abst['type'] == 'mask':
            masks = [action[2] for action in actions]
            ab = Abstractor(masks)
            self.encoder = ab.get_encoder()

            self.actions = []
            n_cells = len(actions[0][0])*len(actions[0][0][0])
            z = 0
            post = np.array(self.encoder.predict(np.reshape(actions[0][0],[-1,n_cells]))[0][0])
            
            reshaping = np.reshape(masks,[len(masks),len(masks[0])*len(masks[0][0])]) 
            predictions = self.encoder.predict(reshaping)

            for i in range(len(actions)):
                if i % 100 == 0:         
                    print("{}-esima azione tradotta".format(i))
                              
                pre = post
                post = np.array(predictions[0][i])
                self.actions += [np.array([pre,actions[i][1],post])]


            print(self.actions[0])
        
        elif config.abst['type'] == 'filtered_mask':
            masks = [actions[i][2] for i in range(len(actions))]
            ab = Abstractor(masks)
            self.encoder = ab.get_encoder()

            self.actions = []
            n_cells = len(actions[0][0])*len(actions[0][0][0])
            z = 0
            post = np.array(self.encoder.predict(np.reshape(actions[0][0],[-1,n_cells]))[0][0])

            reshaping = np.reshape(masks,[len(masks),len(masks[0])*len(masks[0][0])]) 
            predictions = self.encoder.predict(reshaping)

            for i in range(len(actions)):
                if i % 100 == 0:         
                    print("{}-esima azione tradotta".format(i))
                              
                pre = post
                post = np.array(predictions[0][i])
                self.actions += [np.array([pre,actions[i][1],post])]


            print(self.actions[0])

        elif config.abst['type'] == 'image':
            images = [actions[i][2] for i in range(len(actions))]
            print("Total images {} for VAE and BS".format(len(images)))

            cbsm = cv2.createBackgroundSubtractorMOG2(len(images)) 
            for i in range(len(images)): 
                cbsm.apply(images[i][2],images[i][0])

            self.background = cbsm.getBackgroundImage()
            images = np.average(abs(images - self.background),axis=3) != 0
            
            ab = Abstractor(images)
            self.encoder = ab.get_encoder()

            self.actions = []
            n_cells = len(actions[0][0])*len(actions[0][0][0])

            post = np.array(self.encoder.predict(np.reshape(np.average(abs(actions[0][0] - self.background),axis=2) != 0,[-1,n_cells]))[0][0])

            reshaping = np.reshape(images,[len(images),len(images[0])*len(images[0][0])]) 
            predictions = self.encoder.predict(reshaping)
            

            
            for i in range(len(actions)):
                if i % 100 == 0:         
                    print("{}-esima azione tradotta".format(i))
                              
                pre = post
                post = np.array(predictions[0][i])
                self.actions += [np.array([pre,actions[i][1],post])]


            print(self.actions[0])

        else:
            self.actions = actions

        self.dictionary_abstract_actions = {}

        #For each variable in actions condition it add a list to put the significative differences 
        condition_dimension = len(self.actions[0][0])
        self.lists_significative_differences = [[] for i in range(condition_dimension)]

        ordered_differences_queues = [pq.PriorityQueue() for i in range(condition_dimension)]   

        differences = abs(np.take(self.actions,0,axis=1)-np.take(self.actions,2,axis=1))
        for i in range(condition_dimension): 
            for j in range(len(self.actions)): 
                ordered_differences_queues[i].enqueue(None, differences[j][i])         

        actions_to_remove = int(np.floor(len(self.actions)*config.abst['percentage_of_actions_ignored_at_the_extremes']))
        
        for i in range(condition_dimension): 
            sup = ordered_differences_queues[i].get_queue_values()
            for j in np.linspace(actions_to_remove,len(self.actions)-1-actions_to_remove, config.abst['total_abstraction']).round(0):
                self.lists_significative_differences[i] += [sup[int(j)]]
            

    def get_abstraction(self, abstraction_level):
        '''
        Calculate the vector representing the abstraction required as input in each variable

        Args:
            abstraction_level (int)

        Returns:
            distances (numpy.ndarray): where the i-th cell of the vector represents the input abstraction on the i-th variable         
        '''    
        return np.array([self.lists_significative_differences[i][abstraction_level] for i in range(len(self.lists_significative_differences))])

    def get_dist(self, cond1, cond2):
        '''
        Calculate the amount of abstractions that distances the two input conditions

        Args:
            cond1 (list of float)
            cond2 (list of float)

        Returns:
            distance (int)         
        '''      
        return np.sum(abs(cond1-cond2))  

    def get_encoder(self):
        return self.encoder 

    def background_subtractor(self, img):
        return abs(img - self.background)

