mkdir -p ~/rpm
yumdownloader --destdir ~/rpm --resolve mesa-libOSMesa.x86_64 mesa-libOSMesa-devel.x86_64 patchelf.x86_64
cd ~/rpm
for rpm in `ls`; do rpm2cpio $rpm | cpio -id ; done
