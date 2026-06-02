# Face-Region-Classifier
# Face Region Classification using Deep Learning

顔画像を入力として、深層学習（CNN）を用いて人物を4つの地域（アジア、アフリカ、中東、ヨーロッパ）に分類する画像分類モデルです。

## 概要

本プロジェクトでは、畳み込みニューラルネットワーク（CNN）を用いて顔画像の特徴を学習し、4地域の分類を行います。画像は複数の畳み込み層とプーリング層を通して特徴抽出され、その後の全結合層によって最終的な分類結果を出力します。

## 分類カテゴリ

- Asia
- Africa
- Middle East
- Europe

## ネットワーク構成

- Convolution Layer × 4
- ReLU Activation × 6
- Max Pooling × 2
- Fully Connected Layer × 3
- Dropout × 2
- Softmax + Cross Entropy Loss

## 学習設定

| 項目 | 値 |
|------|------|
| Epochs | 60 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Number of Classes | 4 |
| Input Size | 32 × 32 × 3 |

## 使用技術

- Python
- NumPy
- Matplotlib
- Deep Learning
- Convolutional Neural Network (CNN)

## 学習手順

1. 顔画像データの読み込み
2. CNNによる特徴抽出
3. Forward Propagation
4. Back Propagation
5. Adamによるパラメータ更新
6. 精度評価
7. モデル保存（`param.pkl`）

## 出力

入力された顔画像に対して、以下のいずれかの地域カテゴリを予測します。

- asia
- africa
- tyuutou
- euro

## データセット

本プロジェクトでは、顔画像データを用いて4地域（Asia、Africa、Middle East、Europe）の分類を行います。

データセットはリポジトリに含まれていません。利用者自身で顔画像を収集し、各カテゴリにラベル付けを行った上でデータセットを作成する必要があります。

学習データは NumPy 形式（`.npz`）で保存し、以下の配列を含むことを想定しています。

- `images` : 顔画像データ
- `labels` : 教師ラベル

例：

```python
npz = np.load("images_data.npz")

X = npz["images"]   # 入力画像
T = npz["labels"]   # 教師ラベル
```

## データ前処理

学習前に以下の前処理を実施します。

### 画像データ

- 配列形式を `(batch, height, width, channel)` から `(batch, channel, height, width)` に変換
- `float16` 型へ変換
- 画素値を 0〜1 に正規化

```python
X = X.transpose(0, 3, 1, 2).astype(np.float16)
X /= 255.
```

### ラベルデータ

教師ラベルを One-Hot Encoding に変換します。

```python
T = np.eye(len(c_names))[T].astype(np.float16)
```

### 学習データと検証データへの分割

```python
split = 192

x_train = X[:split]
t_train = T[:split]

x_valid = X[split:]
t_valid = T[split:]
```

本実装では、データセットを学習用データと検証用データに分割し、モデルの学習と性能評価を行います。

## 注意

顔画像データには著作権やプライバシーに関する問題が含まれる可能性があります。データセットを作成する際は、利用条件や権利関係を十分に確認してください。
本モデルの性能は学習データセットの品質や構成に依存します。分類結果は学習データの偏りやサンプル数の影響を受ける可能性があります。
