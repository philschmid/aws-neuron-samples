# Configure Linux for Neuron repository updates
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Update OS packages
sudo apt-get update -y
# Install OS headers
sudo apt-get install linux-headers-$(uname -r) -y
# Install Neuron Driver
sudo apt-get install aws-neuronx-dkms -y
# Install Neuron Tools
sudo apt-get install aws-neuronx-tools -y

export PATH=/opt/aws/neuron/bin:$PATH

pip install -U pip

# Set Pip repository  to point to the Neuron repository
pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
#Install Neuron PyTorch
pip install torch-neuron neuron-cc[tensorflow] "protobuf==3.20.1" torchvision transformers