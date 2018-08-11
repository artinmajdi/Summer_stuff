from tqdm import tnrange, tqdm_notebook
from time import sleep

for i in tqdm_notebook(5, desc='1st loop'):
    for j in tqdm_notebook(5, desc='2nd loop'):

        # for k in trange(10, desc='3nd loop'):
        sleep(0.01)
100
