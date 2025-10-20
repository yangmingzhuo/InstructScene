# conda create -n instructscene python=3.9
# conda activate instructscene

# pip3 install -i https://download.pytorch.org/whl/cu121 -U torch==2.1.2 torchvision==0.17.1 torchaudio==2.1.2
pip3 install -r settings/requirements.txt

python3 -c "import nltk; nltk.download('cmudict')"
