DATA_PATH=/data/imagenet/ilsvrc2012
MODEL_PATH=/home/shinryu/caffe/imagenet/alexnet

python  test_recognition.py \
    --testset      $DATA_PATH/val.txt  \
    --basepath     $DATA_PATH/val  \
    --mean         $MODEL_PATH/ilsvrc_2012_mean.npy \
    --prototxt     $MODEL_PATH/deploy.prototxt  \
    --caffemodel   $MODEL_PATH/bvlc_alexnet.caffemodel  \
    --gpu_id       0
