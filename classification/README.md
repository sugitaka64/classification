# テキストデータ加工 クラスタリング処理

## 機能

### scripts/svm.py

サポートベクターマシン（SVM）でクラスタリングを行う。  
カーネル関数はRBFカーネルを使用。  
また、グリッドサーチも実行する。（検証するパラメータはコード内部にベタ書き。）  
cf. http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

## 使用方法

### 入力データのフォーマット

* ヘッダ行無しのCSV形式であること
* 1列目が目的変数（クラスタ番号）、2列目以降が説明変数（特徴量）になっていること

### 出力されるファイルのフォーマット

* ヘッダ行無しのCSV形式
* 1列目が実際の目的変数、2列目が予測された目的変数（特徴量）

### scripts/svm.py

```
$ python /path/to/text_processinag/classification/scripts/svm.py \
    --input_file_path=<input_file_path>
    --output_file_path=<output_file_path>
```

* input_file_path: クラスタデータ（CSV形式）
* output_file_path: 予測データ（CSV形式）の保存先

