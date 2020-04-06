# Feel free to share, modify, copy. I retain no rights to this code. -James Whitehead
# Let me know if you have any questions at james.whitehead490@gmail.com

# This code was tested using python 2.7

# This program is a demonstration of how to perform parallel operations on numpy array elements
# While python can perform concurrent operations, it cannot share work across multiple CPUs (parallel). To do this,
# we must use a library written in a language than can.
# For this example, we'll use the multiprocessing library to apply a pool of worker threads.

import numpy as np
import multiprocessing as mp
import time

# Some constants
PRINT_VALUES = False  # Flag to print the values being processed. Can make the output pretty messy.
THREAD_SLEEP_TIME = 1.  # Time each thread to sleep in seconds

# First, we need to define a function that will be applied to each element in the array. In this example, we'll do a
# simple element-wise array multiplication
def do_work(arguments):
    """
    arguments:
        A dictionary of input variables
        Avoid list/tuple/ndarray arguments since numpy will unpack it into the array. I recommend using a dictionary like in
        this example
        The function can only have a single argument (for map to work)
    """

    # have the thread sleep for THREAD_SLEEP_TIME so it's clear we are executing in parallel
    time.sleep(THREAD_SLEEP_TIME)

    # Perform some work so we know the thread has executed
    return arguments['a'] * arguments['b']

if __name__=='__main__':
    # We make an input array for the pool to consume.
    # For this example, we'll make a 3D array

    # Feel free to mess with the size and dimension. It should still work. You might want to change THREAD_SLEEP_TIME
    # so it doesn't take too long
    shape = (2, 2, 3)
    a = np.random.rand(*shape)
    b = np.random.rand(*shape)

    # Creates a ndarray with the same shape as a/b. The returned array contains a dictionary at each element.
    # The np.vectorize decorator forces this function to take each element of the ndarray instead of the array as
    # a whole
    @np.vectorize
    def build_argument_array(a, b):
        return dict([('a', a), ('b', b)])

    argument_array = build_argument_array(a, b)

    if PRINT_VALUES: print("argument_array:\n{}".format(argument_array))


    # There is a problem with Pool that it only can process 1-D lists of inputs and outputs. To be able to work with
    # N-d arrays, we can reshape the N-d array into a 1-D array before the Pool and then reshape the 1-D back into the
    # N-d
    # To do this, we'll use the numpy.{reshape, ravel_multi_index, unravel_index}

    #create a flat array of inputs
    argument_array_flat = np.reshape(argument_array, np.prod(shape))
    if PRINT_VALUES: print("argument_array_flat: \n{}".format(argument_array_flat))

    # Setup parallel pool
    p = mp.Pool() # The number of workers will default to the number of CPUs

    #prints some stats
    print("Number of CPUs: {}".format(mp.cpu_count()))
    print("Number of array elements: {}".format(np.prod(shape)))
    print("Per-thread sleep time: {}".format(THREAD_SLEEP_TIME))
    print("Total sleep time: {}".format(np.prod(shape) * THREAD_SLEEP_TIME))

    # Get start time for benchmarking
    t_0 = time.time()

    # Pool's map function is similar to python's map. In fact, when debugging parallel code, it's helpful to replace
    # p.map with map to get more informative error messages.
    results_flat = p.map(do_work, argument_array_flat)
    if PRINT_VALUES: print("results_flat: \n{}".format(results_flat))

    # Now the pool has done the work, we are given a flat array of results. The last thing to do is to reshape the
    # results into the input shape

    results = np.reshape(results_flat, shape)
    if PRINT_VALUES: print("results: \n{}".format(results))

    print("Execution took {} seconds".format(time.time() - t_0))

    # And finally, we verify that the result is what we expected
    assert(np.allclose(results, a*b))
    print("Arrays have the same value!!!")

