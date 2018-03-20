from __future__ import print_function
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import time
from PIL import Image
import numpy as np

from keras import backend
from keras.models import Model
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

content_image_path = './data/wife2.jpeg'
style_image_path = './data/sea.png'

cols, rows = load_img(content_image_path).size
height = 400
width = int(cols / rows * height)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(height, width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

content_image = backend.variable(preprocess_image(content_image_path))
style_image = backend.variable(preprocess_image(style_image_path))
combination_image = backend.placeholder((1, height, width,3))

input_tensor = backend.concatenate([content_image, style_image, combination_image], axis=0)

model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

layers = dict([(layer.name, layer.output) for layer in model.layers])
for layer in model.layers:
    print(layer.name)

content_weight = 0.025
style_weight = 20.0
total_variation_weight = 1.0

loss = backend.variable(0.)

def content_loss(content, combination):
    return backend.sum(backend.square(combination-content))

layer_features = layers['block2_conv2']
content_image_features = layer_features[0,:,:,:]
combination_features = layer_features[2,:,:,:]

loss += content_weight * content_loss(content_image_features, combination_features)

def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2,0,1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(S-C)) / (4.*(channels**2)*(size**2))

feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv4', 'block4_conv4',
                  'block5_conv4']
for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1,:,:,:]
    combination_features = layer_features[2,:,:,:]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

def total_variation_loss(x):
    a = backend.square(x[:,:height-1,:width-1,:] - x[:,1:,:width-1,:])
    b = backend.square(x[:,:height-1,:width-1,:] - x[:,:height-1,1:,:])
    return backend.sum(backend.pow(a+b, 1.25))

loss += total_variation_weight*total_variation_loss(combination_image)

grads = backend.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = backend.function([combination_image], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1,height,width,3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

x = preprocess_image(content_image_path)

iterations = 15

for i in range(iterations):
    print('start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time-start_time))

x = x.reshape((height, width,3))
x[:,:,0] += 103.939
x[:,:,1] += 116.779
x[:,:,2] += 123.68
x = x[:,:,::-1]
x = np.clip(x, 0, 255).astype('uint8')

imsave('./result/outfile.jpeg', x)
