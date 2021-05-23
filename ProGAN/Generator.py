import numpy as np
import tensorflow as tf

def pixel_norm(x, epsilon=1e-8):
  with tf.variable_scope('PixelNorm'):
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)
  
def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
  if fan_in is None: fan_in = np.prod(shape[:-1])
  std = gain / np.sqrt(fan_in) # He init
  if use_wscale:
    wscale = tf.constant(np.float32(std), name='wscale')
    return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
  else:
    return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
  
def upscale2d_conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
  assert kernel >= 1 and kernel % 2 == 1
  w = get_weight([kernel, kernel, fmaps, x.shape[1].value], gain=gain, use_wscale=use_wscale, fan_in=(kernel**2)*x.shape[1].value)
  w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
  w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
  w = tf.cast(w, x.dtype)
  os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
  return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

def conv2d(x, fmaps, kernel, gain=np.sqrt(2), use_wscale=False):
  assert kernel >= 1 and kernel % 2 == 1
  w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
  w = tf.cast(w, x.dtype)
  return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

def upscale2d(x, factor=2):
  assert isinstance(factor, int) and factor >= 1
  if factor == 1: return x
  with tf.variable_scope('Upscale2D'):
    s = x.shape
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = tf.tile(x, [1, 1, 1, factor, 1, factor])
    x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x

def leaky_relu(x, alpha=0.2):
  with tf.name_scope('LeakyRelu'):
    alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
    return tf.maximum(x * alpha, x)
  
def dense(x, fmaps, gain=np.sqrt(2), use_wscale=False):
  if len(x.shape) > 2:
    x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
  w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
  w = tf.cast(w, x.dtype)
  return tf.matmul(x, w)
  
def apply_bias(x):
  b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros())
  b = tf.cast(b, x.dtype)
  if len(x.shape) == 2:
    return x + b
  else:
    return x + tf.reshape(b, [1, -1, 1, 1])
      
def Generator( latents_in, labels_in, num_channels = 1, resolution = 32, label_size = 0,
  fmap_base   = 8192,         # Overall multiplier for the number of feature maps.
  fmap_decay  = 1.0,          # log2 feature map reduction when doubling the resolution.
  fmap_max    = 512,          # Maximum number of feature maps in any layer.
  latent_size = None,         # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).     
  use_pixelnorm = True, pixelnorm_epsilon = 1e-8, dtype = 'float32',
  fused_scale = True,         # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
  **kwargs):                         
    
  resolution_log2 = int(np.log2(resolution))
  assert resolution == 2**resolution_log2 and resolution >= 4
    
  def nf(stage):
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    
  def PN(x):
    if use_pixelnorm:
      return pixel_norm(x)
    else :
      return x
      
  if latent_size is None:
    latent_size = nf(0)
    
  latents_in.set_shape([None, latent_size])
  labels_in.set_shape([None, label_size])
  combo_in = tf.cast(tf.concat([latents_in, labels_in], axis=1), dtype)
  lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0.0), trainable=False), dtype)

  # Building blocks.
  def block(x, res): # res = 2..resolution_log2
    with tf.variable_scope('%dx%d' % (2**res, 2**res)):
      if res == 2: # 4x4
        x = pixel_norm(x)
        with tf.variable_scope('Dense'):
          x = dense(x, fmaps=nf(res-1)*16, gain=np.sqrt(2)/4, use_wscale=True)
          x = tf.reshape(x, [-1, nf(res-1), 4, 4])
          x = PN(leaky_relu(apply_bias(x)))
        with tf.variable_scope('Conv'):
          x = PN(leaky_relu(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=True))))        
      else: # 8x8 and up
        if fused_scale:
          with tf.variable_scope('Conv0_up'):
            x = PN(leaky_relu(apply_bias(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=True))))
        else:
          x = upscale2d(x)
          with tf.variable_scope('Conv0'):
            x = PN(leaky_relu(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=True))))
          with tf.variable_scope('Conv1'):
            x = PN(leaky_relu(apply_bias(conv2d(x, fmaps=nf(res-1), kernel=3, use_wscale=True))))
                    
      return x
          
  def torgb(x, res): # res = 2..resolution_log2
    lod = resolution_log2 - res
    with tf.variable_scope('ToRGB_lod%d' % lod):
      return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=True))

  # Recursive structure: complex but efficient.
  def grow(x, res, lod):
    y = block(x, res)
    img = lambda: upscale2d(torgb(y, res), 2**lod)
    if res > 2: 
      img = cset(img, (lod_in > lod), lambda: upscale2d(lerp(torgb(y, res), upscale2d(torgb(x, res - 1)), lod_in - lod), 2**lod))
    if lod > 0: 
      img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
        
    return img()
    
  images_out = grow(combo_in, 2, resolution_log2 - 2)
        
  assert images_out.dtype == tf.as_dtype(dtype)
  images_out = tf.identity(images_out, name='images_out')
    
  return images_out
