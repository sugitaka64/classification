# テキストデータ加工 デンドログラム描画処理

## 機能

### scripts/dendrogram.py

デンドログラム描画処理を行う。  
cf. https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html

## 使用方法

### 入力データのフォーマット

* ヘッダ行無しのCSV形式であること
* 1列目が目的変数（ラベル、ID・・・等）、2列目以降が説明変数（特徴量）になっていること

### scripts/dendrogram.py

```
$ python /path/to/text_processing/visualization/scripts/dendrogram.py \
    --input_file_path=<input_file_path> \
    [--display_node_num=<display_node_num>] \
    [--x_size=<x_size>] \
    [--y_size=<y_size>]
```

* input_file_path : クラスタリングする入力データ（CSV形式）
* display_node_num: デンドログラム上に表示するノード数（デフォルト: 自動表示）
* x_size          : 表示する図の横サイズ（デフォルト: 8）
* y_size          : 表示する図の縦サイズ（デフォルト: 6）

