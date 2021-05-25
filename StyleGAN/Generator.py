import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

def blur2d(x, f=[1,2,1], normalize=True):
    with tf.variable_scope('Blur2D'):
        @tf.custom_gradient
        def func(x):
            y = _blur2d(x, f, normalize)
            @tf.custom_gradient
            def grad(dy):
                dx = _blur2d(dy, f, normalize, flip=True)
                return dx, lambda ddx: _blur2d(ddx, f, normalize)
            return y, grad
          
        return func(x)

def upscale2d(x, factor=2):
    with tf.variable_scope('Upscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _upscale2d(x, factor)
            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, factor, gain=factor**2)
                return dx, lambda ddx: _upscale2d(ddx, factor)
            return y, grad
          
        return func(x)

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, lrmul=1):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    
    return tf.get_variable('weight', shape=shape, initializer=init) * runtime_coef

def dense(x, fmaps, **kwargs):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    
    return tf.matmul(x, w)

def conv2d(x, fmaps, kernel, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')

def upscale2d_conv2d(x, fmaps, kernel, fused_scale='auto', **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    assert fused_scale in [True, False, 'auto']
    if fused_scale == 'auto':
        fused_scale = min(x.shape[2:]) * 2 >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        return conv2d(upscale2d(x), fmaps, kernel, **kwargs)

    # Fused => perform both ops simultaneously using tf.nn.conv2d_transpose().
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)
    w = tf.transpose(w, [0, 1, 3, 2]) # [kernel, kernel, fmaps_out, fmaps_in]
    w = tf.pad(w, [[1,1], [1,1], [0,0], [0,0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    
    return tf.nn.conv2d_transpose(x, w, os, strides=[1,1,2,2], padding='SAME', data_format='NCHW')

def apply_bias(x, lrmul=1):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1])

def leaky_relu(x, alpha=0.2):
    with tf.variable_scope('LeakyReLU'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        @tf.custom_gradient
        def func(x):
            y = tf.maximum(x, x * alpha)
            @tf.custom_gradient
            def grad(dy):
                dx = tf.where(y >= 0, dy, dy * alpha)
                return dx, lambda ddx: tf.where(y >= 0, ddx, ddx * alpha)
            return y, grad
          
        return func(x)

def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

def instance_norm(x, epsilon=1e-8):
    assert len(x.shape) == 4 # NCHW
    with tf.variable_scope('InstanceNorm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=[2,3], keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2,3], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
        
        return x

def style_mod(x, dlatent, **kwargs):
    with tf.variable_scope('StyleMod'):
        style = apply_bias(dense(dlatent, fmaps=x.shape[1]*2, gain=1, **kwargs))
        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        
        return x * (style[:,0] + 1) + style[:,1]

def apply_noise(x, noise_var=None, randomize_noise=True):
    assert len(x.shape) == 4 # NCHW
    with tf.variable_scope('Noise'):
        if noise_var is None or randomize_noise:
            noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        else:
            noise = tf.cast(noise_var, x.dtype)
        weight = tf.get_variable('weight', shape=[x.shape[1].value], initializer=tf.initializers.zeros())
        
        return x + noise * tf.reshape(tf.cast(weight, x.dtype), [1, -1, 1, 1])
      
def G_style( latents_in, labels_in,
    truncation_psi          = 0.7,                  # Style strength multiplier for the truncation trick. None = disable.
    truncation_cutoff       = 8,                    # Number of layers for which to apply the truncation trick. None = disable.
    truncation_psi_val      = None,                 # Value for truncation_psi to use during validation.
    truncation_cutoff_val   = None,                 # Value for truncation_cutoff to use during validation.
    dlatent_avg_beta        = 0.995,                # Decay for tracking the moving average of W during training. None = disable.
    style_mixing_prob       = 0.9,                  # Probability of mixing styles during training. None = disable.
    is_training             = False,                # Network is under training? Enables and disables specific features.
    is_validation           = False,                # Network is under validation? Chooses which value to use for truncation_psi.
    is_template_graph       = False,                # True = template graph constructed by the Network class, False = actual evaluation.
    components              = dnnlib.EasyDict(),    # Container for sub-networks. Retained between calls.
    **kwargs):

    # Validate arguments.
    assert not is_training or not is_validation
    assert isinstance(components, dnnlib.EasyDict)
    if is_validation:
        truncation_psi = truncation_psi_val
        truncation_cutoff = truncation_cutoff_val
    if is_training or (truncation_psi is not None and not tflib.is_tf_expression(truncation_psi) and truncation_psi == 1):
        truncation_psi = None
    if is_training or (truncation_cutoff is not None and not tflib.is_tf_expression(truncation_cutoff) and truncation_cutoff <= 0):
        truncation_cutoff = None
    if not is_training or (dlatent_avg_beta is not None and not tflib.is_tf_expression(dlatent_avg_beta) and dlatent_avg_beta == 1):
        dlatent_avg_beta = None
    if not is_training or (style_mixing_prob is not None and not tflib.is_tf_expression(style_mixing_prob) and style_mixing_prob <= 0):
        style_mixing_prob = None

    # Setup components.
    if 'synthesis' not in components:
        components.synthesis = tflib.Network('G_synthesis', func_name=G_synthesis, **kwargs)
    num_layers = components.synthesis.input_shape[1]
    dlatent_size = components.synthesis.input_shape[2]
    if 'mapping' not in components:
        components.mapping = tflib.Network('G_mapping', func_name=G_mapping, dlatent_broadcast=num_layers, **kwargs)

    # Setup variables.
    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)
    dlatent_avg = tf.get_variable('dlatent_avg', shape=[dlatent_size], initializer=tf.initializers.zeros(), trainable=False)

    # Evaluate mapping network.
    dlatents = components.mapping.get_output_for(latents_in, labels_in, **kwargs)

    # Update moving average of W.
    if dlatent_avg_beta is not None:
        with tf.variable_scope('DlatentAvg'):
            batch_avg = tf.reduce_mean(dlatents[:, 0], axis=0)
            update_op = tf.assign(dlatent_avg, tflib.lerp(batch_avg, dlatent_avg, dlatent_avg_beta))
            with tf.control_dependencies([update_op]):
                dlatents = tf.identity(dlatents)

    # Perform style mixing regularization.
    if style_mixing_prob is not None:
        with tf.name_scope('StyleMix'):
            latents2 = tf.random_normal(tf.shape(latents_in))
            dlatents2 = components.mapping.get_output_for(latents2, labels_in, **kwargs)
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            cur_layers = num_layers - tf.cast(lod_in, tf.int32) * 2
            mixing_cutoff = tf.cond(
                tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
                lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
                lambda: cur_layers)
            dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)

    # Apply truncation trick.
    if truncation_psi is not None and truncation_cutoff is not None:
        with tf.variable_scope('Truncation'):
            layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_idx.shape, dtype=np.float32)
            coefs = tf.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
            dlatents = tflib.lerp(dlatent_avg, dlatents, coefs)

    # Evaluate synthesis network.
    with tf.control_dependencies([tf.assign(components.synthesis.find_var('lod'), lod_in)]):
        images_out = components.synthesis.get_output_for(dlatents, force_clean_graph=is_template_graph, **kwargs)
        
    return tf.identity(images_out, name='images_out')

def G_mapping( latents_in, labels_in,                          
    latent_size             = 512,          # Latent vector (Z) dimensionality.
    label_size              = 0,            # Label dimensionality, 0 if no labels.
    dlatent_size            = 512,          # Disentangled latent (W) dimensionality.
    dlatent_broadcast       = None,         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
    mapping_layers          = 8,            # Number of mapping layers.
    mapping_fmaps           = 512,          # Number of activations in the mapping layers.
    mapping_lrmul           = 0.01,         # Learning rate multiplier for the mapping layers.
    mapping_nonlinearity    = 'lrelu',
    dtype                   = 'float32',
    **_kwargs): 

    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[mapping_nonlinearity]

    # Inputs.
    latents_in.set_shape([None, latent_size])
    labels_in.set_shape([None, label_size])
    latents_in = tf.cast(latents_in, dtype)
    labels_in = tf.cast(labels_in, dtype)
    x = latents_in

    # Embed labels and concatenate them with latents.
    if label_size:
        with tf.variable_scope('LabelConcat'):
            w = tf.get_variable('weight', shape=[label_size, latent_size], initializer=tf.initializers.random_normal())
            y = tf.matmul(labels_in, tf.cast(w, dtype))
            x = tf.concat([x, y], axis=1)

    # Normalize latents.
    x = pixel_norm(x)

    # Mapping layers.
    for layer_idx in range(mapping_layers):
        with tf.variable_scope('Dense%d' % layer_idx):
            fmaps = dlatent_size if layer_idx == mapping_layers - 1 else mapping_fmaps
            x = dense(x, fmaps=fmaps, gain=gain, use_wscale=True, lrmul=mapping_lrmul)
            x = apply_bias(x, lrmul=mapping_lrmul)
            x = act(x)

    # Broadcast.
    if dlatent_broadcast is not None:
        with tf.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert x.dtype == tf.as_dtype(dtype)
    
    return tf.identity(x, name='dlatents_out')

def G_synthesis(
    dlatents_in,                        
    dlatent_size        = 512,          # Disentangled latent (W) dimensionality.
    num_channels        = 3,            
    resolution          = 1024,         # Output resolution.
    fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
    fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
    fmap_max            = 512,          # Maximum number of feature maps in any layer.      
    const_input_layer   = True,      
    use_noise           = True,        
    randomize_noise     = True,
    nonlinearity        = 'lrelu',     
    use_pixel_norm      = False,    
    use_instance_norm   = True,   
    dtype               = 'float32',
    fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
    blur_filter         = [1,2,1],      # Low-pass filter to apply when resampling activations. None = no filtering.
    structure           = 'recursive',
    **_kwargs):                     

    resolution_log2 = int(np.log2(resolution))
    assert resolution == 2**resolution_log2 and resolution >= 4
    
    def nf(stage): 
      return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
    
    def blur(x): 
      if blur_filter:
        return blur2d(x, blur_filter)
      else: 
        return x

    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[nonlinearity]
    num_layers = resolution_log2 * 2 - 2
    num_styles = num_layers
    images_out = None

    # Primary inputs.
    dlatents_in.set_shape([None, num_styles, dlatent_size])
    dlatents_in = tf.cast(dlatents_in, dtype)
    lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)

    # Noise inputs.
    noise_inputs = []
    if use_noise:
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, use_noise, 2**res, 2**res]
            noise_inputs.append(tf.get_variable('noise%d' % layer_idx, shape=shape, initializer=tf.initializers.random_normal(), trainable=False))

    # Things to do at the end of each layer.
    def layer_epilogue(x, layer_idx):
        if use_noise:
            x = apply_noise(x, noise_inputs[layer_idx], randomize_noise=randomize_noise)
        x = apply_bias(x)
        x = act(x)
        if use_pixel_norm:
            x = pixel_norm(x)
        if use_instance_norm:
            x = instance_norm(x)
        if use_styles:
            x = style_mod(x, dlatents_in[:, layer_idx], use_wscale=True)
            
        return x

    # Early layers.
    with tf.variable_scope('4x4'):
        if const_input_layer:
            with tf.variable_scope('Const'):
                x = tf.get_variable('const', shape=[1, nf(1), 4, 4], initializer=tf.initializers.ones())
                x = layer_epilogue(tf.tile(tf.cast(x, dtype), [tf.shape(dlatents_in)[0], 1, 1, 1]), 0)
        else:
            with tf.variable_scope('Dense'):
                x = dense(dlatents_in[:, 0], fmaps=nf(1)*16, gain=gain/4, use_wscale=True) # tweak gain to match the official implementation of Progressing GAN
                x = layer_epilogue(tf.reshape(x, [-1, nf(1), 4, 4]), 0)
        with tf.variable_scope('Conv'):
            x = layer_epilogue(conv2d(x, fmaps=nf(1), kernel=3, gain=gain, use_wscale=True), 1)

    # Building blocks for remaining layers.
    def block(res, x): # res = 3..resolution_log2
        with tf.variable_scope('%dx%d' % (2**res, 2**res)):
            with tf.variable_scope('Conv0_up'):
                x = layer_epilogue(blur(upscale2d_conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=True, fused_scale=fused_scale)), res*2-4)
            with tf.variable_scope('Conv1'):
                x = layer_epilogue(conv2d(x, fmaps=nf(res-1), kernel=3, gain=gain, use_wscale=True), res*2-3)
            
            return x
          
    def torgb(res, x): # res = 2..resolution_log2
        lod = resolution_log2 - res
        with tf.variable_scope('ToRGB_lod%d' % lod):
            return apply_bias(conv2d(x, fmaps=num_channels, kernel=1, gain=1, use_wscale=True))

    # Fixed structure: simple and efficient, but does not support progressive growing.
    if structure == 'fixed':
        for res in range(3, resolution_log2 + 1):
            x = block(res, x)
        images_out = torgb(resolution_log2, x)

    # Recursive structure: complex but efficient.
    if structure == 'recursive':
        def cset(cur_lambda, new_cond, new_lambda):
            return lambda: tf.cond(new_cond, new_lambda, cur_lambda)
          
        def grow(x, res, lod):
            y = block(res, x)
            img = lambda: upscale2d(torgb(res, y), 2**lod)
            img = cset(img, (lod_in > lod), lambda: upscale2d(tflib.lerp(torgb(res, y), upscale2d(torgb(res - 1, x)), lod_in - lod), 2**lod))
            if lod > 0: img = cset(img, (lod_in < lod), lambda: grow(y, res + 1, lod - 1))
              
            return img()
          
        images_out = grow(x, 3, resolution_log2 - 3)

    assert images_out.dtype == tf.as_dtype(dtype)
    
    return tf.identity(images_out, name='images_out')      
