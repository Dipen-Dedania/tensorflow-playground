import tensorflow as tf

# A feature that is convenient to use when we want to explore the data content of an object is
# tf.InteractiveSession(). Using it and .eval() method, we can get a full look at the values
# without the need to constantly refer to the session object

# Initialize two constants
x1 = tf.constant([1,2,3])
print(x1.get_shape())

x2 = tf.constant([4,5,6])
print(x2.get_shape())

result = tf.multiply(x1, x2)
sess = tf.InteractiveSession()
print('multiply result: \n {}'.format(result.eval()))

###########################################################
# Now we will try with the matrix

x3 = tf.constant([[1,2,3],[3,2,1]])
print(x3.get_shape())

x4 = tf.constant([1,0,2])
print(x4.get_shape())

x4 = tf.expand_dims(x4,1)
print(x4.get_shape())

result2 = tf.matmul(x3, x4)
sess = tf.InteractiveSession()
print('After expanding x4 dimension: \n {}'.format(x4.eval()))
print('matmul result2: \n {}'.format(result2.eval()))
