import horovod.keras as hvd

hvd.init()

hvd.local_rank()
