# アーキテクチャ詳細

## 計算フロー

### Phase 1: 定常状態計算

```mermaid
graph LR
    A[パラメータ設定] --> B[グリッド生成]
    B --> C[政策関数初期化]
    C --> D{EGM反復}
    D -->|収束| E[分布計算]
    D -->|未収束| C
    E --> F{分布収束}
    F -->|収束| G[定常状態完了]
    F -->|未収束| E
```

### Phase 2: SSJによるヤコビアン計算

```mermaid
graph LR
    A[定常状態] --> B[Dual EGM]
    B --> C[政策微分計算]
    C --> D[Fake News]
    D --> E[分布摂動]
    E --> F[Toeplitz行列構築]
    F --> G[ブロックヤコビアン]
```

### Phase 3: 一般均衡

```mermaid
graph LR
    A[ショック定義] --> B[部分均衡応答]
    B --> C[GE乗数計算]
    C --> D[不平等分析]
    D --> E[可視化出力]
```

---

## コンポーネント詳細

### MultiDimGrid

3次元状態空間 $(m, a, z)$ のインデックス管理:

```cpp
// フラットインデックス ↔ 座標変換
idx = iz * (N_a * N_m) + ia * N_m + im
```

### TwoAssetSolver

EGMの2段階最適化:
1. **No-Adjust**: $a' = a(1+r_a)$ を固定して流動性資産のみ最適化
2. **Adjust**: 非流動性資産の調整を許可（調整コスト考慮）

### JacobianBuilder3D

Dual数による自動微分:
- $r_m$ に対する感応度: 予算制約経由
- $r_a$ に対する感応度: 資産成長経由
- $w$ に対する感応度: 労働所得経由

### FakeNewsAggregator

分布の1次摂動:
$$\partial D_{t+1} = \Lambda \cdot \partial D_t + F \cdot \partial \theta_t$$

### GeneralEquilibrium

市場清算条件による閉鎖:
$$dY = (I - J_{CY})^{-1} J_{Cr} \cdot dr$$

---

## パフォーマンス

| グリッドサイズ | 政策収束 | 分布収束 | 合計時間 |
|---------------|---------|---------|---------|
| 50×40×2       | ~500 iter | ~1000 iter | ~30s |
| 100×80×5      | ~600 iter | ~1500 iter | ~180s |

---

## 拡張ポイント

1. **New Keynesian ブロック**: 価格硬直性、フィリップス曲線
2. **財政政策**: 累進課税、給付金
3. **失業リスク**: 2状態所得過程
4. **資本蓄積**: 企業セクター
