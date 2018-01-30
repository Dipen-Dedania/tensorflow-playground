# The result of the lines of code is an abstract tensor in the computation graph.
# However, contrary to what you might expect, the result doesn’t actually get calculated;
# It just defined the model but no process ran to calculate the result.
# You can see this in the print-out: there’s not really a result that you want to see (namely, 30).
# This means that TensorFlow has a lazy evaluation!

import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Print the result
print(result)

