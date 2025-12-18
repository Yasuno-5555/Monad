# API リファレンス

## C++ コアAPI

### TwoAssetParam

モデルパラメータを保持する構造体。

```cpp
struct TwoAssetParam {
    double beta;    // 割引因子 (0.97)
    double sigma;   // CRRA係数 (2.0)
    double r_m;     // 流動性資産金利 (0.01)
    double r_a;     // 非流動性資産金利 (0.05)
    double chi;     // 調整コストスケール (20.0)
    double m_min;   // 借入下限 (-2.0)
};
```

### MultiDimGrid

3次元グリッド管理クラス。

```cpp
class MultiDimGrid {
public:
    UnifiedGrid m_grid;  // 流動性資産グリッド
    UnifiedGrid a_grid;  // 非流動性資産グリッド
    int n_z;             // 所得状態数
    
    int idx(int im, int ia, int iz);           // 座標→インデックス
    void get_coords(int flat, int& im, int& ia, int& iz);  // インデックス→座標
    std::pair<double, double> get_values(int flat);        // 値取得
};
```

### TwoAssetSolver

政策関数ソルバー。

```cpp
class TwoAssetSolver {
public:
    TwoAssetSolver(const MultiDimGrid& grid, const TwoAssetParam& params);
    
    // 1ステップBellman反復 (EGM)
    double solve_bellman(const TwoAssetPolicy& current, 
                         TwoAssetPolicy& next,
                         const IncomeProcess& income);
    
    // 定常状態の期待値（公開）
    std::vector<double> E_Vm_next;
    std::vector<double> E_V_next;
};
```

### SsjSolver3D

Sequence Space Jacobianソルバー。

```cpp
class SsjSolver3D {
public:
    SsjSolver3D(const MultiDimGrid& grid, 
                const TwoAssetParam& params,
                const IncomeProcess& income,
                const TwoAssetPolicy& pol_ss,
                const std::vector<double>& D_ss,
                const std::vector<double>& E_Vm,
                const std::vector<double>& E_V);
    
    // ブロックヤコビアン計算
    // 戻り値: Output -> Input -> Matrix(T×T)
    std::map<std::string, std::map<std::string, Eigen::MatrixXd>> 
        compute_block_jacobians(int T);
};
```

### GeneralEquilibrium

一般均衡ソルバー。

```cpp
class GeneralEquilibrium {
public:
    GeneralEquilibrium(SsjSolver3D& solver, int horizon);
    
    // 金融政策ショックのGE解
    std::map<std::string, Eigen::VectorXd> 
        solve_monetary_shock(const Eigen::VectorXd& dr_m);
};
```

### InequalityAnalyzer

不平等分析クラス。

```cpp
class InequalityAnalyzer {
public:
    InequalityAnalyzer(const MultiDimGrid& grid,
                       const std::vector<double>& D_ss,
                       const TwoAssetPolicy& pol_ss,
                       const PartialMap& partials);
    
    struct GroupPaths {
        Eigen::VectorXd top10;
        Eigen::VectorXd bottom50;
        Eigen::VectorXd debtors;
    };
    
    GroupPaths analyze_consumption_response(
        const std::map<std::string, Eigen::VectorXd>& dU_paths);
    
    std::vector<double> compute_consumption_heatmap(
        const std::map<std::string, Eigen::VectorXd>& dU_paths, int t);
};
```

---

## Python API

### vis_inequality.py

```python
def plot_inequality_analysis():
    """
    irf_groups.csv と heatmap_sensitivity.csv を読み込み、
    可視化を生成する。
    
    出力:
        inequality_winners_losers.png
        inequality_heatmap.png
    """
```

---

## 出力ファイル形式

### policy_2asset.csv

| Column | Description |
|--------|-------------|
| m_idx, a_idx, z_idx | グリッドインデックス |
| m_val, a_val, z_val | 状態変数値 |
| c | 消費政策 |
| m_prime | 次期流動性資産 |
| a_prime | 次期非流動性資産 |
| adjust_flag | 調整フラグ (1=調整) |

### ge_irf.csv

| Column | Description |
|--------|-------------|
| t | 時点 |
| dr_m | 金利ショック |
| dY | 産出変化 |
| dC | 消費変化 |
| dC_direct | 直接効果 |
| dC_indirect | 間接効果 (GE) |

### irf_groups.csv

| Column | Description |
|--------|-------------|
| time | 時点 |
| top10 | Top 10%消費変化 |
| bottom50 | Bottom 50%消費変化 |
| debtors | 債務者消費変化 |
| aggregate | 総消費変化 |
