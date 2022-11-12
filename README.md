# rock5b-wav2letter-bench
rock5b running armnn wav2letter GpuAcc &amp; CpuAcc exaples
```
sudo apt update 
sudo apt upgrade
sudo apt install python3-dev python3-pip python3-venv git git-lfs libsndfile1 libportaudio2 software-properties-common clinfo ocl-icd-libopencl1 libmali-valhall-g610-g6p0-x11
git clone https://github.com/StuartIanNaylor/rock5b-wav2letter-bench.git
cd rock5b-wav2letter-bench
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
pip3 install --upgrade pip

sudo bash -c 'echo "libmali-valhall-g610-g6p0-x11.so" > /etc/OpenCL/vendors/mali.icd'
git clone https://github.com/ARM-software/armnn/

```

