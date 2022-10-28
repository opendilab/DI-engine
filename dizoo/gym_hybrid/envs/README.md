# Modified gym-hybrid

The gym-hybrid directory is modified from https://github.com/thomashirtz/gym-hybrid.     
We add the HardMove environment additionally.  (Please refer to https://arxiv.org/abs/2109.05490 Section 5.1 for details about HardMove env.) 

Specifically, the modified gym-hybrid contains the following three types of environments:

- Moving-v0 
- Sliding-v0
- HardMove-v0 

### Install Guide

```bash
cd DI-engine/dizoo/gym_hybrid/envs/gym-hybrid
pip install -e .
```

## Acknowledgement

https://github.com/thomashirtz/gym-hybrid