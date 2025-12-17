# 使用ガイド

## 基本的な使い方

### 1. パラメータ設定

`src/main_two_asset.cpp` でパラメータを調整:

```cpp
Monad::TwoAssetParam params;
params.beta = 0.97;    // 割引因子
params.r_m = 0.01;     // 流動性資産金利 (年率1%)
params.r_a = 0.05;     // 非流動性資産金利 (年率5%)
params.chi = 20.0;     // 調整コスト
params.sigma = 2.0;    // CRRA
params.m_min = -2.0;   // 借入下限
```

### 2. グリッド設定

```cpp
// 流動性資産: [-2, 50], 50点
auto m_grid = make_grid(50, -2.0, 50.0, 3.0);

// 非流動性資産: [0, 100], 40点
auto a_grid = make_grid(40, 0.0, 100.0, 2.0);

// 所得過程: 2状態
auto income = make_income();
```

### 3. 実行

```bash
# ビルド
g++ -std=c++17 src/main_two_asset.cpp -I . -I build_phase3/_deps/eigen-src/ -o MonadTwoAsset.exe

# 実行
./MonadTwoAsset.exe
```

### 4. 出力確認

```bash
# CSVファイル確認
head policy_2asset.csv
head ge_irf.csv
head irf_groups.csv

# 可視化
python monad/vis_inequality.py
```

---

## カスタマイズ

### 所得過程の変更

```cpp
IncomeProcess make_income() {
    IncomeProcess p;
    p.n_z = 3;  // 3状態に変更
    p.z_grid = {0.5, 1.0, 1.5};  // 低・中・高所得
    p.Pi_flat = {
        0.8, 0.15, 0.05,
        0.1, 0.8,  0.1,
        0.05, 0.15, 0.8
    };
    return p;
}
```

### ショックの変更

```cpp
// AR(1)ショック (持続性0.8)
Eigen::VectorXd dr_m(T);
double rho = 0.8;
double shock_size = 0.01;  // 1%
for(int t=0; t<T; ++t) dr_m[t] = shock_size * std::pow(rho, t);
```

### 分析グループの変更

`InequalityAnalyzer` で追加グループ定義:

```cpp
// 例: 中間層 (50-90パーセンタイル)
auto idx_middle = get_indices_by_wealth_percentile(0.50, 0.90);
```

---

## トラブルシューティング

### 収束しない場合

1. グリッドを粗くする（計算高速化）
2. 初期政策を調整
3. 反復回数上限を増やす

### メモリ不足

1. グリッドサイズ削減
2. スパース行列の使用確認
3. 64bit環境で実行

### 数値不安定

1. `chi` (調整コスト) を増やす
2. 借入下限 `m_min` を緩める
3. 割引因子 `beta` を下げる

---

## 高度な使用法

### Python連携

```python
import subprocess
import pandas as pd

# C++エンジン実行
subprocess.run(["./MonadTwoAsset.exe"])

# 結果読み込み
df = pd.read_csv("ge_irf.csv")
print(f"GE乗数: {df['dY'][0] / df['dC_direct'][0]:.3f}")
```

### バッチ実行

```python
import json

for chi in [10, 20, 50]:
    # パラメータ変更 (要: config対応)
    subprocess.run(["./MonadTwoAsset.exe"])
    df = pd.read_csv("ge_irf.csv")
    print(f"chi={chi}: 乗数={df['dY'][0]/df['dC_direct'][0]:.3f}")
```
