# -*- coding: utf-8 -*-
'''
test_recognition.py
==============================
caffe を対象とした分類問題の評価ツール
'''
from __future__ import print_function
import argparse
import logging
import os

import caffe
import cv2
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--testset', required=True,
        help='Pairs of image and label sample definition text file.'
    )
    parser.add_argument(
        '--basepath', default=('.', '.'),
        help='Path to a image and label samples.'
    )
    parser.add_argument(
        '--mean_file',
        help='Mean image of a train samples.'
    )
    parser.add_argument(
        '--prototxt', required=True,
        help='Model definition file.'
    )
    parser.add_argument(
        '--caffemodel', required=True,
        help='Trained model\'s parameter file.'
    )
    parser.add_argument(
        '--gpu_id', type=int, default=-1,
        help='Using gpu number.'
    )
    return parser.parse_args()

def start_logging():
    '''
    ロギングを開始する。
    表示するログは info 以上。
    '''
    date_format = '%m/%d %H:%M:%S'
    log_format = '[%(asctime)s %(module)s.py:%(lineno)d] '
    log_format += '%(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format
    )

def parse_text(testset, basepath):
    '''
    テキストファイルから1行ずつ読み込んで入力と教師に分離する。

    Arguments
    ------------------------------
    testset : str
        画像ファイルと教師ファイルのペアが記述されているテキストファイル。
    basepath : str
        testset から読み込んだファイルのパス。ファイル名は含まない。

    Returns
    ------------------------------
    data_list : (str, int)
        入力サンプルと教師データのペアのリスト。
    '''
    data_list = []
    with open(testset, 'r') as f:
        for line in f:
            samples = line.replace(',', ' ').split()
            image_file = os.path.join(basepath, samples[0])
            # label が書いてない場合はラベルを -1 として処理する
            label_file = int(samples[1]) if len(samples) == 2 else -1
            data_list.append((image_file, label_file))
    return data_list

def evaluation(net, data_list, mean_file):
    '''
    モデルにテスト画像を入力して推定クラスを得る。

    Arguments
    ------------------------------
    net: caffe._caffe.Net
        caffe.Net() で生成されたモデル
    data_list: ((str, int), ...)
        net に入れるテストデータ
    mean_file: str
        テストサンプルから減算する平均画像
    '''
    mean = np.load(mean_file)
    in_size = net.blobs[net.inputs[0]].data.shape[2]
    blob_shape = net.blobs[net.inputs[0]].data.shape[1:]

    top1_accuracy = 0.0
    top5_accuracy = 0.0
    for image_file, label in data_list:
        x = cv2.imread(image_file, cv2.CV_LOAD_IMAGE_COLOR).astype(np.float32)
        x = preprocess(x, mean, in_size)
        if not x.shape == blob_shape:
            raise Exception('Not match shape of the blob and input: {} vs {}'.format(blob_shape, x.shape))
        x = x[np.newaxis, :, :, :]
        prob = net.forward_all(**{net.inputs[0]:x})[net.outputs[0]][0]
        pred = np.argsort(prob)[-5:][::-1]
        assert len(pred) == 5

        top1_accuracy += label == pred[0]
        top5_accuracy += label in pred

        msg = '{}: [{:3d}, {:3d}, {:3d}, {:3d}, {:3d}]'.format(
              os.path.basename(image_file), pred[0], pred[1], pred[2], pred[3], pred[4])
        print(msg)
    print('Top1 accuracy: {}'.format(top1_accuracy / len(data_list)))
    print('Top5 accuracy: {}'.format(top5_accuracy / len(data_list)))

def preprocess(x, mean, in_size):
    '''
    ILSVRC で使われている画像の切り出し方法。
    サンプルの短辺を target_size(一般的には256)になるように
    アスペクト比を維持してリサイズ。
    リサイズしたサンプルの中心を in_size x in_size になるように切り出す。

    Arguments
    ------------------------------
    x: numpy.ndarray
        テストサンプル
    mean: numpy.ndarray
        平均画像
    in_size: int
        モデルに入力するサイズ
    '''
    target_size = 256
    xh, xw = x.shape[:2]
    out_h = target_size * xh / xw if xw < xh else target_size
    out_w = target_size * xw / xh if xw > xh else target_size
    assert in_size <= out_h
    assert in_size <= out_w
    x = cv2.resize(x, (out_w, out_h))

    start_h = (out_h - target_size) // 2
    start_w = (out_w - target_size) // 2
    stop_h = start_h + target_size
    stop_w = start_w + target_size
    x = x[start_h:stop_h, start_w:stop_w, :]

    start = (target_size - in_size) // 2
    stop = start + in_size
    x = x[start:stop, start:stop, :].transpose(2, 0, 1)
    mean = mean[:, start:stop, start:stop]
    return x - mean

def main():
    args = parse_arguments()
    start_logging()

    if not os.path.isfile(args.prototxt):
        raise IOError('Not found a prototxt: {}'.format(args.prototxt))
    if not os.path.isfile(args.caffemodel):
        raise IOError('Not found a caffemodel: {}'.format(args.caffemodel))

    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    if 0 <= args.gpu_id:
        logging.info('Gpu mode. Using gpu device id: {}.'.format(args.gpu_id))
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()
    else:
        logging.info('Cpu mode.')
        caffe.set_mode_cpu()

    data_list = parse_text(args.testset, args.basepath)
    logging.info('Loading testset: {}'.format(args.testset))
    logging.info('Loading prototxt: {}'.format(args.prototxt))
    logging.info('Loading caffemodel: {}'.format(args.caffemodel))
    logging.info('# of test samples: {}'.format(len(data_list)))

    logging.info('Evaluating...')
    evaluation(net, data_list, args.mean_file)

if __name__ == '__main__':
    main()
