## deps
#### 安装 MuJoCo
1. 获取通行证 [MuJoCo website](https://www.roboti.us/license.html). 

* 在 [MuJoCo website](https://www.roboti.us/license.html) 上下载一个名为 `getid_linux` 的运行文件
* 在你想要运行 mujoco 的机器上运行上面这个文件，得到一串密文
* 在 [MuJoCo website](https://www.roboti.us/license.html) 上输入密文和你的用户名以及邮箱，然后会得到一个包含 `mjkey.txt` 的邮件
* 将 `mjkey.txt` 置于 `~/.mujoco/` 下
* 下载环境 `wget https://www.roboti.us/download/mujoco200_linux.zip`，解压并将里面的东西放到 `~/.mujoco/mujoco200` 下
* 通过 `pip install --user mujoco_py==2.0.2.8` 安装模块

**注意，`mjkey.txt` 是和运行节点绑定的。比如说你要在 SH-IDC1-10-198-8-236 上面运行 mujoco，那么你需要在这台机器上面 `srun -p vi_x_cerebra_meta -w SH-IDC1-10-198-8-236 ./getid_linux` 获取到对应这个机器的密文。（由于srun有缓存机制，需要按一下enter退出后正在运行的二进制文件 `getid_linux` 后，才会显示机器密文）**

**./addition 目录下面有 SH-IDC1-10-198-8-236 对应的 `mjkey.txt`，有效期到 2020.1.17**

## 测试 
```
$ export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH
$ python
Python 3.6.9 |Anaconda, Inc.| (default, Jul 30 2019, 19:07:31)
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import mujoco_py
```

如果不报错，那就说明安装成功

## 可能的问题
1. 在集群管理节点可以import，但是srun之后不能import，而且报错 `fatal error: GL/osmesa.h: No such file or directory`

解决方法：

```
$ export LD_LIBRARY_PATH=/mnt/lustre/zhangming/.mujoco/mujoco200/bin:/lib64:$LD_LIBRARY_PATH 
$ sh ./addition/install_mesa.sh
$ export PATH="$PATH:$HOME/rpm/usr/bin"
$ export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/rpm/usr/lib:$HOME/rpm/usr/lib64"
$ export LDFLAGS="-L$HOME/rpm/usr/lib -L$HOME/rpm/usr/lib64"
$ export CPATH="$CPATH:$HOME/rpm/usr/include"
```
然后重新开一个窗口，或者在 `LD_LIBRARY_PATH` 中把 `/lib64` 去掉，然后重新运行


