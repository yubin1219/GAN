import numpy as np
import tensorflow as tf

def lerp(a, b, t):
  return a + (b - a) * t

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
  if fan_in is None: fan_in = np.prod(shape[:-1])
  std = gain / np.sqrt(fan_in) 
  if use_wscale:
    wscale = tf.constant(np.float32(std), name='wscale')
    return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
  else:
    return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
  if len(x.shape) > 2:
    x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
  w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
  w = tf.cast(w, x.dtype)
  return tf.matmul(x, w)

def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
  assert kernel >= 1 and kernel % 2 == 1
  w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
  w = tf.cast(w, x.dtype)
  return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

def leaky_relu(x, alpha=0.2):
  with tf.name_scope('LeakyRelu'):
    alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
    return tf.maximum(x * alpha, x)

def apply_bias(x):
  b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
  b = tf.cast(b, x.dtype)
  if len(x.shape) == 2:
    return x + b
  else:
    return x + tf.reshape(b, [1, -1, 1, 1])

def downscale2d(x, factor=2):
  assert isinstance(factor, int) and factor >= 1
  if factor == 1: return x
  with tf.variable_scope('Downscale2D'):
    ksize = [1, 1, factor, factor]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW')
  
def conv2d_downscale2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
  assert kernel >= 1 and kernel % 2 == 1
  w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
  w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
  w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
  w = tf.cast(w, x.dtype)
  return tf.nn.conv2d(x, w, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

def minibatch_stddev_layer(x, group_size=4):
  with tf.variable_scope('MinibatchStddev'):
    group_size = tf.minimum(group_size, tf.shape(x)[0])     
    s = x.shape                                            
    y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])  
    y = tf.cast(y, tf.float32)      
    y -= tf.reduce_mean(y, axis=0, keepdims=True)    
    y = tf.reduce_mean(tf.square(y), axis=0)       
    y = tf.sqrt(y + 1e-8)                              
    y = tf.reduce_mean(y, axis=[1,2,3], keepdims=True)  
    y = tf.cast(y, x.dtype)                              
    y = tf.tile(y, [group_size, 1, s[2], s[3]])        
    return tf.concat([x, y], axis=1)   
      
def Discriminator( images_in, num_channels = 1, resolution = 32, label_size = 0,
  fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
  fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
  fmap_max            = 512,          # Maximum number of feature maps in any layer. 
  mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
  dtype               = 'float32',    # Data type to use for activations and outputs.
  fused_scale         = True,         # True = use fused conv2d + downscale2d, False = separate downscale2d layers.
  **kwargs):                          # Ignore unrecognized keyword args.
    
  resolution_log2 = int(np.log2(resolution))
  assert resolution == 2**resolution_log2 and resolution >= 4
  def nf(stage): 
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

  images_in.set_shape([None, num_channels, resolution, resolution])
  images_in = tf.cast(images_in, dtype)
  lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

  # Building blocks.
  def fromrgb(x, res): # res = 2..resolution_log2
    with tf.variable_scope('FromRGB_lod%d' % (resolution_log2 - res)):
      return leaky_relu(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=1, use_wscale=True)))
    
  def block(x, res): # res = 2..resolution_log2
    with tf.variable_scope('%dx%d' % (2**res, 2**res)):
      if res >= 3: # 8x8 and up
        with tf.variable_scope('Conv0'):
          x = leaky_relu(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=True)))
        if fused_scale:
          with tf.variable_scope('Conv1_down'):
            x = leaky_relu(apply_bias(conv2d_downscale2d(x, fmaps=nf(res-2), kernel=3, use_wscale=True)))
        else:
          with tf.variable_scope('Conv1'):
            x = leaky_relu(apply_bias(conv2d(x, fmaps=nf(res-2), kernel=3, use_wscale=True)))
          x = downscale2d(x)
      else: # 4x4
        if mbstd_group_size > 1:
          x = minibatch_stddev_layer(x, mbstd_group_size)
        with tf.variable_scope('Conv'):
          x = leaky_relu(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=True)))
        with tf.variable_scope('Dense0'):
          x = leaky_relu(apply_bias(dense(x, fmaps=nf(res-2), use_wscale=True)))
        with tf.variable_scope('Dense1'):
          x = apply_bias(dense(x, fmaps=1+label_size, gain=1, use_wscale=True))
      return x

  # Recursive structure: complex but efficient.
  def grow(res, lod):
    x = lambda: fromrgb(downscale2d(images_in, 2**lod), res)
    if lod > 0: 
      x = cset(x, (lod_in < lod), lambda: grow(res + 1, lod - 1))
    x = block(x(), res)
    y = lambda: x
    if res > 2: 
      y = cset(y, (lod_in > lod), lambda: lerp(x, fromrgb(downscale2d(images_in, 2**(lod+1)), res - 1), lod_in - lod))
    return y()
    
  combo_out = grow(2, resolution_log2 - 2)

  assert combo_out.dtype == tf.as_dtype(dtype)
  scores_out = tf.identity(combo_out[:, :1], name='scores_out')
  labels_out = tf.identity(combo_out[:, 1:], name='labels_out')
  
  return scores_out, labels_out
