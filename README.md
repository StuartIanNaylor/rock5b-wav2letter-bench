# rock5b-wav2letter-bench
rock5b running armnn wav2letter GpuAcc &amp; CpuAcc exaples
```
sudo apt update 
sudo apt upgrade
sudo apt install python3-dev python3-pip python3-venv git git-lfs libsndfile1 libportaudio2 software-properties-common clinfo ocl-icd-libopencl1 libmali-valhall-g610-g6p0-x11 ffmpeg
git lfs install --skip-repo
git clone https://github.com/StuartIanNaylor/rock5b-wav2letter-bench.git
cd rock5b-wav2letter-bench
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
pip3 install --upgrade pip

sudo bash -c 'echo "libmali-valhall-g610-g6p0-x11.so" > /etc/OpenCL/vendors/mali.icd'
sudo add-apt-repository ppa:armnn/ppa
sudo apt update
sudo apt-get install -y python3-pyarmnn armnn-latest-all

git clone https://github.com/ARM-software/armnn/
cd armnn/python/pyarmnn/examples/speech_recognition
pip3 install -r requirements.txt
cd ~/rock5b-wav2letter-bench
git clone https://github.com/ARM-software/ML-zoo
cd armnn/python/pyarmnn/examples/speech_recognition
cp -r ~/rock5b-wav2letter-bench/ML-zoo/models/speech_recognition/wav2letter/tflite_int8 .
mkdir -p samples
wget --quiet --show-progress -O samples/gb0.ogg https://upload.wikimedia.org/wikipedia/commons/2/22/George_W._Bush%27s_weekly_radio_address_%28November_1%2C_2008%29.oga
wget --quiet --show-progress -O samples/gb1.ogg https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W_Bush_Columbia_FINAL.ogg
wget --quiet --show-progress -O samples/hp0.ogg https://upload.wikimedia.org/wikipedia/en/d/d4/En.henryfphillips.ogg
wget --quiet --show-progress -O samples/mm1.wav https://cdn.openai.com/whisper/draft-20220913a/micro-machines.wav
wget --quiet --show-progress -O samples/quick_brown_fox_16000khz.wav https://git.mlplatform.org/ml/armnn.git/tree/python/pyarmnn/examples/speech_recognition/tests/testdata/quick_brown_fox_16000khz.wav
echo "Converting to 16-bit WAV ..."
ffmpeg -loglevel -0 -y -i samples/gb0.ogg -ar 16000 -ac 1 -c:a pcm_s16le samples/gb0.wav

ffmpeg -loglevel -0 -y -i samples/gb1.ogg -ar 16000 -ac 1 -c:a pcm_s16le samples/gb1.wav

ffmpeg -loglevel -0 -y -i samples/hp0.ogg -ar 16000 -ac 1 -c:a pcm_s16le samples/hp0.wav

ffmpeg -loglevel -0 -y -i samples/mm1.wav -ar 16000 -ac 1 -c:a pcm_s16le samples/mm0.wav

rm samples/mm1.wav

python3 run_audio_file.py --audio_file_path tests/testdata/quick_brown_fox_16000khz.wav --model_file_path tflite_int8/wav2letter_int8.tflite --preferred_backends CpuAcc CpuRef
```
What you can do is run on the GPU GpuAcc and choose one of the much bigger samples
`python3 run_audio_file.py --audio_file_path samples/gb0.wav --model_file_path tflite_int8/wav2letter_int8.tflite --preferred_backends GpuAcc CpuRef`
But the original run_audio_file.py uses a pretty horrendous python based MFCC routine for each audio chunk which makes it impossible to differentiate Cpu load vs GPU

```
cp ~/rock5b-wav2letter-bench/preload_mfcc_run_audio_file.py .
pip3 install psutil
python3 preload_mfcc_run_audio_file.py --audio_file_path samples/gb0.wav --model_file_path tflite_int8/wav2letter_int8.tflite --preferred_backends GpuAcc CpuRef
```
The audio will take time to load and pause on `Processing Audio Frames...`
To change back to CpuAcc simply change
`python3 preload_mfcc_run_audio_file.py --audio_file_path samples/gb0.wav --model_file_path tflite_int8/wav2letter_int8.tflite --preferred_backends CpuAcc CpuRef`
CpuAcc is Neon optimised, GpuAcc is Mali Optimised and CpuRef is just a single thread simple cpu fallback



