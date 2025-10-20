TAG=$1
DEVICE=$2

CUDA_VISIBLE_DEVICES=$DEVICE \
python3 -m src.train_objfeatvqvae configs/threedfront_objfeat_vqvae.yaml \
  --tag $TAG
