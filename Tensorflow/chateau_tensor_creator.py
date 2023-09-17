import os

import numpy as np

import einops

import tensorflow as tf
from keras import layers

absolute_path = os.path.dirname(__file__)


#checks shape of tensors
class ShapeChecker():
  def __init__(self):
    # Keep a cache of every axis-name seen
    self.shapes = {}

  def __call__(self, tensor, names, broadcast=False):
    if not tf.executing_eagerly():
      return

    parsed = einops.parse_shape(tensor, names)

    for name, new_dim in parsed.items():
      old_dim = self.shapes.get(name, None)
      
      if (broadcast and new_dim == 1):
        continue

      if old_dim is None:
        # If the axis name is new, add its length to the cache.
        self.shapes[name] = new_dim
        continue

      if new_dim != old_dim:
        raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                         f"    found: {new_dim}\n"
                         f"    expected: {old_dim}\n")



relative_path = "../coworker raw.csv"
path_to_file = os.path.join(absolute_path, relative_path)

#Emilia,23/07/21,July,23/08/21,17,Secondary,Fluent,Female,None,0,31

#returns 2 numpy arrays, one of all english sentences, one of german ones
def load_data(path):
    file = open(path, "r")
    text = file.read()
    lines = text.splitlines()[1:]
    sets = [line.split(',') for line in lines]
    modified_sets = []
    for set in sets:
        modified_sets += [[[int(set[10])],[set[2], int(set[4]), set[5], set[6], set[7], set[8]]]]
    context = np.array([context for target, context in modified_sets])
    target = np.array([target for target, context in modified_sets])
    file.close()

    return target, context

target_raw, context_raw = load_data(path_to_file)


#from here

BUFFER_SIZE = len(context_raw) #number of sentences
BATCH_SIZE = 64

is_train = np.random.uniform(size=(len(target_raw),)) < 0.8

#train set
train_raw = (
    tf.data.Dataset
    .from_tensor_slices((context_raw[is_train], target_raw[is_train]))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE))

#test set
val_raw = (
    tf.data.Dataset
    .from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE))




#to here
#this takes the english and german sentence tensors, puts corresponding sentences together, creates
#a np array of random booleans, 20% of which are false
#uses that array to select sentences from the tensors when creating datasets, which is just a data
#format, it randomises their order and splits them in to arrays of length 64




#strips weird characters, whitespace, adds start and end caps, puts spaces on sides of punctuation,
#puts all to lower case
def normaliser(data):
    month_vocab = ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"]
    gender_vocab = ["Male", "Female"]
    language_vocab = ["Basic", "Intermediate", "Fluent"]
    bar_vocab = ["Waiter", "Bar"]
    school_vocab = ["None", "Secondary", "College"]
    vocab = month_vocab+gender_vocab+language_vocab+bar_vocab+school_vocab
    layer = layers.StringLookup(vocabulary=vocab)
    vectorized_data = layer(data)
    return vectorized_data





#uses above processors to convert text to numbers, targ in and targ out are same, just shifted forward by one
def process_text(context, target):
  context = normaliser(context)
  #target = normaliser(target)
  return context, target

#converts the training and test sets into the integer values with maps


train_ds = train_raw.map(process_text)
val_ds = val_raw.map(process_text)

#done processing input data, now the model

model = tf.keras.Sequential(name="Shithole")
model.add(layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2), loss='mse')
model.fit(train_ds, validation_data=val_ds, epochs=1000)

input_text = "18,July,Secondary,Fluent,Female,None"
def inputProcessor(input):
  return normaliser(np.array([input.split(',')]))

input_text = inputProcessor(input_text)



def outputProcessor(output):
  val = abs(output[0][0])
  #while val>400:
  #  val /= 10
  return np.round(val)

print(outputProcessor(model.predict(input_text)))

model.save(absolute_path)

print("DONE")