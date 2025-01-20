import tensorflow as tf

training_data_dir = 'dataset'

# Load Data
data = tf.keras.utils.image_dataset_from_directory(training_data_dir, batch_size=32, image_size=(512,512), color_mode='grayscale')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Scale Data
data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()

# Split Data
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)