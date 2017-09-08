# テキストデータ加工 次元削減処理

## 機能

### scripts/svd.py

SVD（特異値分解）で次元削減を行う。  
cf. http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

### scripts/lda.py

LDA（≒ LSA:潜在意味解析）で次元削減を行う。  
cf. https://radimrehurek.com/gensim/models/ldamodel.html


### scripts/lsi.py

LSI（潜在的意味インデキシング）で次元削減を行う。  
<font color="red">※指定した次元数未満になることがあるので注意。（原因調査中）</font>  
cf. https://radimrehurek.com/gensim/models/lsimodel.html

### scripts/plsa.py

PLSA（確率的潜在意味解析）で次元削減を行う。  
cf. http://qiita.com/HZama/items/0957f74f8da1302f7652  
    http://qiita.com/HZama/items/561cb240620991d3a0e2

### scripts/pca.py

PCA（主成分分析）で次元削減を行う。  
また、引数で指定した次元数での累積寄与率を出力する。  
cf. http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

## 使用方法

### 入力ファイルのフォーマット

* ヘッダ行無しのCSV形式であること
* 1列目が目的変数（ラベル、ID・・・等）、2列目以降が説明変数（特徴量）になっていること

### 出力されるファイルのフォーマット

* ヘッダ行無しのCSV形式
* 1列目が目的変数（ラベル、ID・・・等）、2列目以降が説明変数（特徴量）

### scripts/svd.py

```
$ python /path/to/text_processing/dimensionality_reduction/scripts/svd.py \
    --input_file_path=<input_file_path> \
    --output_file_path=<output_file_path> \
    --num_topics=<num_topics>
```

* input_file_path : 次元削減前の入力ファイル（CSV形式）
* output_file_path: 次元削減後の出力ファイル（CSV形式）
* num_topics      : 削減後の次元数

### scripts/lda.py

```
$ python /path/to/text_processing/dimensionality_reduction/scripts/lda.py \
    --input_file_path=<input_file_path> \
    --output_file_path=<output_file_path> \
    --num_topics=<num_topics>
```

* input_file_path : 次元削減前の入力ファイル（CSV形式）
* output_file_path: 次元削減後の出力ファイル（CSV形式）
* num_topics      : 削減後の次元数

### scripts/lsi.py

```
$ python /path/to/text_processing/dimensionality_reduction/scripts/lsi.py \
    --input_file_path=<input_file_path> \
    --output_file_path=<output_file_path> \
    --num_topics=<num_topics>
```

* input_file_path : 次元削減前の入力ファイル（CSV形式）
* output_file_path: 次元削減後の出力ファイル（CSV形式）
* num_topics      : 削減後の次元数

### scripts/plsa.py

```
$ python /path/to/text_processing/dimensionality_reduction/scripts/plsa.py \
    --input_file_path=<input_file_path> \
    --output_file_path=<output_file_path> \
    --num_topics=<num_topics>
```

* input_file_path : 次元削減前の入力ファイル（CSV形式）
* output_file_path: 次元削減後の出力ファイル（CSV形式）
* num_topics      : 削減後の次元数

### scripts/pca.py

```
$ python /path/to/text_processing/dimensionality_reduction/scripts/pca.py \
    --input_file_path=<input_file_path> \
    --output_file_path=<output_file_path> \
    --num_topics=<num_topics>
```

* input_file_path : 次元削減前の入力ファイル（CSV形式）
* output_file_path: 次元削減後の出力ファイル（CSV形式）
* num_topics      : 削減後の次元数

