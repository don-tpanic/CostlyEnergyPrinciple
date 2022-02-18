import tensorflow as tf


class NoisyOnes(tf.keras.initializers.Initializer):
    """
    Weights initializer: ones +/- noise.
    """
    def __init__(self, noise_level, noise_distribution, random_seed):
        self.noise_level = noise_level
        self.noise_distribution = noise_distribution
        self.random_seed = random_seed

    def __call__(self, shape, dtype=None):
        
        ones = tf.ones(
            shape=shape, dtype=tf.dtypes.float32, name=None
        )
        
        if self.noise_distribution == 'uniform':
            
            noise = tf.random.uniform(
                shape=shape, 
                minval=-self.noise_level, 
                maxval=self.noise_level, 
                dtype=tf.dtypes.float32, 
                seed=self.random_seed, 
                name=None
            )
        else:
            NotImplementedError()
            
        noisy_ones = ones + noise
        # print(f'max={tf.reduce_max(noisy_ones)}, min={tf.reduce_min(noisy_ones)}')
        return noisy_ones    
        
    def get_config(self):  # To support serialization
        return {
          'noise_level': self.noise_level, 
          'noise_distribution': self.noise_distribution,
          'random_seed': self.random_seed
        }