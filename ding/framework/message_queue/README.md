# Notes on using torchrpc

## Problems you may encounter

Message queue of Torchrpc uses [tensorpipe](https://github.com/pytorch/tensorpipe) as a communication backend, a high-performance modular tensor-p2p communication library. However, several tensorpipe defects have been found in the test, which may make it difficult for you to use it.

### 1. container environment

Tensorpipe is not container aware. Processes can find themselves on the same physical machine through `/proc/sys/kernel/random/boot_id`, but because in separated pod/container, they cannot use means of communication such as CUDA ipc. When tensorpipe finds that these communication methods cannot be used, it will report an error and exit. 

### 2. RDMA and fork subprocess

Tensorpipe does not consider the case of calling [fork(2)](https://man7.org/linux/man-pages/man2/fork.2.html) when using RDMA. If the corresponding initialization measures are not performed when using RDMA, using fork will cause serious problems, refer to [here](https://www.rdmamojo.com/2012/05/24/ibv_fork_init/). Therefore, if you start ditask in the IB/RoCE network environment, please specify the environment variables `IBV_FORK_SAFE=1` and `RDMAV_FORK_SAFE=1` , so that ibverbs will automatically initialize fork support.