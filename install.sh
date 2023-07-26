
echo "Clean the old build."
rm -rf build

mkdir datasets

conda create -n refinebox python=3.10 -y \
&& conda activate refinebox \
&& conda install -c "nvidia/label/cuda-11.7.0" cuda-nvcc -y \
&& conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y \
&& git clone https://gitlab.com/yiqunchen1999/detectron2.git \
&& python -m pip install -e detectron2 \
&& git clone https://gitlab.com/yiqunchen1999/detrex.git \
&& python -m pip install -e detrex \
&& git submodule init \
&& git submodule update \
&& python -m pip install -e . \
&& echo "Rename folder 'checkpoints' to output." \
&& mv checkpoints output \
&& echo "Installation finished, please link your data under the folder datasets."
