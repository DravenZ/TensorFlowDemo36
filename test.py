# encoding: utf-8

"""
__author__ = "Zhang Pengfei"
__date__ = 2018/10/30
"""
import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
# with tf.Session() as sess:
#     with tf.device("/cpu:0"):
#         # print(sess.run(hello))
#
#         matrix1 = tf.constant([[3., 3.]])
#         matrix2 = tf.constant([[2.], [2.]])
#         product = tf.matmul(matrix1, matrix2)
#         print(sess.run(product))


# 创建一个变量, 初始化为标量 0.
# state = tf.Variable(0, name="counter")
#
# # 创建一个 op, 其作用是使 state 增加 1
#
# one = tf.constant(1)
# new_value = tf.add(state, one)
# update = tf.assign(state, new_value)
#
# # 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# # 首先必须增加一个`初始化` op 到图中.
# init_op = tf.initialize_all_variables()
#
# # 启动图, 运行 op
# with tf.Session() as sess:
#     # 运行 'init' op
#     sess.run(init_op)
#     # 打印 'state' 的初始值
#     print(sess.run(state))
#     # 运行 op, 更新 'state', 并打印 'state'
#     for _ in range(3):
#         sess.run(update)
#         print(sess.run(state))
#
# input1 = tf.constant(3.0)
# input2 = tf.constant(2.0)
# input3 = tf.constant(5.0)
# intermed = tf.add(input2, input3)
# mul = tf.multiply(input1, intermed)
#
# with tf.Session() as sess:
#     result = sess.run([mul, intermed])
#     print(result)
#
# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
# output = tf.multiply(input1, input2)
#
# with tf.Session() as sess:
#     print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))

# 下载数据
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


x = tf.placeholder("float", [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)

y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
