# テキストデータ加工 SSE計算処理

## 機能

### scripts/sse.py

ラベル付けされたクラスタのSSE（残差平方和）を計算する。  
cf. https://onlinehelp.tableau.com/current/pro/desktop/ja-jp/clustering_description.html  
※「グループ内の平方和」を参照。

### scripts/mean.py

ラベル付けされたクラスタ毎の平均値を計算する。  
結果はCSVファイルとして出力する。

## 使用方法

### 入力データのフォーマット

* ヘッダ行無しのCSV形式であること
* 1列目が目的変数（クラスタ番号）、2列目以降が説明変数（特徴量）になっていること

### 出力されるファイルのフォーマット

* ヘッダ行無しのCSV形式
* 1列目が目的変数（クラスタ番号）、2列目以降が説明変数（特徴量）

### scripts/sse.py

```
$ python /path/to/text_processing/sse/scripts/sse.py \
    --input_file_path=<input_file_path>
```

* input_file_path: クラスタデータ（CSV形式）

### scripts/mean.py

```
$ python /path/to/text_processing/sse/scripts/mean.py \
    --input_file_path=<input_file_path> \
    --output_file_path=<output_file_path>
```

* input_file_path : クラスタデータ（CSV形式）
* output_file_path: クラスタ毎の平均値データ（CSV形式）
