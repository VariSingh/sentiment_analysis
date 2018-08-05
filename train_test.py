import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np

from sa import create_feature_sets_and_labels


train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')


nodes_layer1 = 500
nodes_layer2 = 500
nodes_layer3 = 500
classes = 2
batch_size = 100
epochs = 10

x = tf.placeholder(tf.float32,[None,len(train_x[0])])
y = tf.placeholder(tf.float32,[None,2])

def neural_network(data):
    hidden_layer1 = {
    'weights':tf.Variable(tf.random_normal([len(train_x[0]),nodes_layer1])),
    'biases':tf.Variable(tf.random_normal([nodes_layer1]))
    }
    hidden_layer2 = {
    'weights':tf.Variable(tf.random_normal([nodes_layer1,nodes_layer2])),
    'biases':tf.Variable(tf.random_normal([nodes_layer2]))
    }
    hidden_layer3 = {
    'weights':tf.Variable(tf.random_normal([nodes_layer2,nodes_layer3])),
    'biases':tf.Variable(tf.random_normal([nodes_layer3]))
    }
    output_layer = {
    'weights':tf.Variable(tf.random_normal([nodes_layer3,classes])),
    'biases':tf.Variable(tf.random_normal([classes]))
    }

    layer1 = tf.add(tf.matmul(data,hidden_layer1['weights']),hidden_layer1['biases'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1,hidden_layer2['weights']),hidden_layer2['biases'])
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2,hidden_layer3['weights']),hidden_layer3['biases'])
    layer3 = tf.nn.relu(layer3)

    output = tf.add(tf.matmul(layer3,output_layer['weights']),output_layer['biases'])
    output = tf.nn.relu(output)

    return output

#predicition
prediction = neural_network(x)
#calculate cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))#we pass prediction and label to compare
#minimize cost
optimizer = tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)


    for epoch in range(epochs):
        loss = 0
        # for _ in range(int(mnist.train.num_examples/batch_size)):
        #     _x,_y = mnist.train.next_batch(batch_size)
        i=0
        while i<len(train_x):
            start = i
            end = i+batch_size

            _x = np.array(train_x[start:end])
            _y = np.array(train_y[start:end])

            _,c = sess.run([optimizer,cost], feed_dict = {x:_x,y:_y})
            loss += c
            i+=batch_size

        print('Epoch ',epoch, 'out of ',epochs, 'loss ',loss)

    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    print('Accuracy: ',accuracy.eval({x:test_x,y:test_y}))

neural_network(x)
