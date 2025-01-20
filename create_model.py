from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from matplotlib import pyplot as plt
from build_data import train, val

# Build Model
model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(512,512,1)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())


model.add(Flatten(input_shape=(512,512,1)))


model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))


model.add(Dense(1, activation='sigmoid'))

# Train Model
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
hist = model.fit(train, epochs=20, validation_data=val)

model.save('model.h5')

# Plot accuracy and loss
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.savefig('figures/Loss_Figure.png')

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.savefig('figures/Accuracy_Figure.png')

