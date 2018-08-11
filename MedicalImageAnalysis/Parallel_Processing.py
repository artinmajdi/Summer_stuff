import multiprocessing as mp
import random
import string
import os

random.seed(123)

# Define an output queue
output = mp.Queue()

subfield = os.listdir('/media/groot/Seagate Backup Plus Drive/')
# define a example function
def rand_string(length, pos, output):
    # """ Generates a random string of numbers, lower- and uppercase chars. """
    # rand_str = ''.join(random.choice(
    #                     string.ascii_lowercase
    #                     + string.ascii_uppercase
    #                     + string.digits)
    #                for i in range(length))
    rand_str = pos
    output.put((pos, rand_str))

# Setup a list of processes that we want to run
processes = [mp.Process(target=rand_string, args=(5, x, output)) for x in subfield]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]

print(results)
