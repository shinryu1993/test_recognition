# test recognition
---
Caffeにおける分類問題の評価ツールです。

## Requirements
本ソースコードは以下のライブラリが必要です。  
- NumPy
- OpenCV
- Caffe

## プログラム引数
- testset: テストサンプルと教師サンプルのペアが書かれたテキストファイル
- basepath: テストサンプルと教師サンプルまでのパス
- mean_file: ImageNetなどで使われる平均画像を指定する
- prototxt: 評価に用いる prototxt
- caffemodel: 評価に用いる caffemodel
- gpu_id: 動作させるGPUのID。CPUで動作させるなら-1を指定する

## 実行例(GPU)
```  
DATA_PATH=/data/imagenet/ilsvrc2012
MODEL_PATH=/home/shinryu/caffe/imagenet/alexnet

python  test_recognition.py \
    --testset      $DATA_PATH/val.txt  \
    --basepath     $DATA_PATH/val  \
    --mean         $MODEL_PATH/ilsvrc_2012_mean.npy \
    --prototxt     $MODEL_PATH/deploy.prototxt  \
    --caffemodel   $MODEL_PATH/bvlc_alexnet.caffemodel  \
    --gpu_id       0
```
