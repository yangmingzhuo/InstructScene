ROOM_TPYE=$1
TAG=$2
DEVICE=$3

CUDA_VISIBLE_DEVICES=$DEVICE \
python3 -m src.train_sg configs/${ROOM_TPYE}_sg_diffusion_vq_objfeat.yaml \
  --tag $TAG
