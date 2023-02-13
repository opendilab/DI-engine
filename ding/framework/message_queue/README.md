# Notes on using torchrpc

## Performance
We conducted performance tests in a k8s environment equipped with A100-80GB and 200G HCA.

### Intra-node GPU-P2P performance

| test case(unit:ms) | 1.25 KB | 20.00 KB | 1.25 MB | 10.00 MB | 40.00 M | 640.00 M | 1.25GB   |
| ------------------ | ------- | -------- | ------- | -------- | ------- | -------- | -------- |
| shm                | 0.3605  | 0.352    | 0.9924  | 7.1229   | 47.9575 | 798.8635 | 1548.782 |
| nccl-nvlink        | 0.1969  | 0.1104   | 0.2162  | 0.3285   | 0.4532  | 3.3166   | 5.3828   |
| cuda-shared-tensor | 0.5307  | 0.578    | 0.9643  | 0.5908   | 1.2449  | 5.3707   | 9.686    |

### Inter-node GPU-P2P performance

| test case(unit:ms)       | 20.00 KB | 1.25 MB | 10.00 MB | 40.00 M  | 640.00 M  | 1.25GB    | 2.50 GB    |
| ------------------------ | -------- | ------- | -------- | -------- | --------- | --------- | ---------- |
| nng-TCP                  | 5.7353   | 9.6782  | 30.5187  | 172.9719 | 3450.7418 | 7083.6372 | 14072.1213 |
| nccl-TCP                 | 0.0826   | 1.321   | 31.7813  | 128.0672 | 1259.72   | 2477.2957 | 5157.7578  |
| nccl-IB                  | 0.0928   | 0.5618  | 2.1134   | 7.1768   | 120.131   | 260.2628  | 518.8091   |
| nccl-GDR (PXN<->PXN)     | 0.5541   | 45.601  | 9.3636   | 19.3071  | 108.11    | 280.0556  | 527.9732   |
| torchrpc-TCP             | 5.6691   | 5.4707  | 14.0155  | 39.4443  | 580.333   | 1154.0793 | 2297.3776  |
| torchrpc-IB              | 21.3884  | 4.4093  | 5.9105   | 22.3012  | 130.249   | 236.8084  | 477.2389   |
| torchrpc-GDR (PXN<->PXN) | 20.5018  | 23.2081 | 15.6427  | 7.5357*  | 48.7812   | 77.2657   | 143.4112   |

### Atari performance
Performance of dizoo/atari/example/atari_dqn_dist_rdma.py
- memory: "32Gi"
- cpu:  16
- gpu: A100


| test case(unit:s) | avg     |
| ----------------- | ------- |
| TCP-nng           | 127.64  |
| torchrpc-CP       | 29.3906 |
| torchrpc-IB       | 28.7763 |


## Problems you may encounter

Message queue of Torchrpc uses [tensorpipe](https://github.com/pytorch/tensorpipe) as a communication backend, a high-performance modular tensor-p2p communication library. However, several tensorpipe defects have been found in the test, which may make it difficult for you to use it.

### 1. container environment

Tensorpipe is not container aware. Processes can find themselves on the same physical machine through `/proc/sys/kernel/random/boot_id` ,but because in separated pod/container, they cannot use means of communication such as CUDA ipc. When tensorpipe finds that these communication methods cannot be used, it will report an error and exit. 

### 2. RDMA and fork subprocess

Tensorpipe does not consider the case of calling [fork(2)](https://man7.org/linux/man-pages/man2/fork.2.html) when using RDMA. If the corresponding initialization measures are not performed when using RDMA, using fork will cause serious problems, refer to [here](https://www.rdmamojo.com/2012/05/24/ibv_fork_init/). Therefore, if you start ditask in the IB/RoCE network environment, please specify the environment variables `IBV_FORK_SAFE=1` and `RDMAV_FORK_SAFE=1` , so that ibverbs will automatically initialize fork support.

### 3. GPU direct RDMA

If you use torchrpc in an environment that supports GPU direct RDMA, if the size of the tensor transmitted in rpc is very small (less than 32B), segmentfault may occur. See [issue.](https://github.com/pytorch/pytorch/issues/57136) We are tracking this bug and hope it can be resolved eventually.
