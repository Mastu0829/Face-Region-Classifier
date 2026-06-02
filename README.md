# Face-Region-Classifier
顔画像を入力として、深層学習（CNN）を用いて人物を4つの地域（アジア、アフリカ、中東、ヨーロッパ）に分類する画像分類モデルです。

概要

本プロジェクトでは、畳み込みニューラルネットワーク（CNN）を用いて顔画像の特徴を学習し、4地域の分類を行います。画像は複数の畳み込み層とプーリング層を通して特徴抽出され、その後全結合層によって最終的な分類結果を出力します。

特徴
CNNによる顔画像分類
4クラス分類
Asia
Africa
Middle East
Europe
ReLU活性化関数
Max Pooling
Dropoutによる過学習抑制
Adamオプティマイザによる学習
ネットワーク構成
Convolution Layer × 4
ReLU Activation × 6
Max Pooling × 2
Fully Connected Layer × 3
Dropout × 2
Softmax + Cross Entropy Loss
学習設定
Epochs: 60
Batch Size: 32
Optimizer: Adam
Learning Rate: 0.001
Number of Classes: 4
入力データ
顔画像
画像サイズ: 32 × 32
RGB画像（3チャンネル）
出力

各顔画像に対して以下のいずれかを予測します。

Asia
Africa
Middle East
Europe
使用技術
Python
NumPy
Matplotlib
Deep Learning
Convolutional Neural Network (CNN)
注意

本モデルは学習データセットの品質や構成に大きく依存します。分類結果は学習データの偏りやサンプル数の影響を受ける可能性があります
