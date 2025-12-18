running build_ext
building 'monad.monad_core' extension
"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\HostX86\x64\cl.exe" /c /nologo /O2 /W3 /GL /DNDEBUG /MD -IC:\Users\kouno\AppData\Local\Programs\Python\Python310\lib\site-packages\pybind11\include -I3rdparty/eigen -Isrc -Isrc/grid -Isrc/kernel -Isrc/solver -Isrc/aggregator -Isrc/ssj -Isrc/blocks -Isrc/analysis -Isrc/experiments -Isrc/io -IC:\Users\kouno\AppData\Local\Programs\Python\Python310\include -IC:\Users\kouno\AppData\Local\Programs\Python\Python310\Include "-IC:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\include" "-IC:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\ATLMFC\include" "-IC:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\VS\include" "-IC:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\ucrt" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.26100.0\\um" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.26100.0\\shared" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.26100.0\\winrt" "-IC:\Program Files (x86)\Windows Kits\10\\include\10.0.26100.0\\cppwinrt" "-IC:\Program Files (x86)\Windows Kits\NETFXSDK\4.8\include\um" /EHsc /Tpsrc/python_bindings.cpp /Fobuild\temp.win-amd64-cpython-310\Release\src\python_bindings.obj /std:c++17 /utf-8 /O2 /EHsc
python_bindings.cpp
C:\Users\kouno\Desktop\Projects\MonadStudio\src\solver/TwoAssetSolver.hpp(660): warning C4244: '初期化中': '__int64' から 'int' への変換です。データが失われる可能性があります。
C:\Users\kouno\Desktop\Projects\MonadStudio\src\solver/TwoAssetSolver.hpp(675): warning C4244: '初期化中': '__int64' から 'int' への変換です。データが失われる可能性があります。
C:\Users\kouno\Desktop\Projects\MonadStudio\src\solver/TwoAssetSolver.hpp(694): warning C4244: '初期化中': '__int64' から 'int' への変換です。データが失われる可能性があります。
C:\Users\kouno\Desktop\Projects\MonadStudio\src\aggregator/DistributionAggregator3D.hpp(176): warning C4244: '=': '__int64' から 'int' への変換です。データが失われる可能性があります。
C:\Users\kouno\Desktop\Projects\MonadStudio\src\ssj\SparseMatrixBuilder.hpp(96): warning C4244: '=': '__int64' から 'int' への変換です。データが失われる可能性があります。
C:\Users\kouno\Desktop\Projects\MonadStudio\src\ssj\JacobianBuilder3D.hpp(84): warning C4244: '初期化中': '__int64' から 'int' への変換です。データが失われる可能性があります。
C:\Users\kouno\Desktop\Projects\MonadStudio\src\ssj\JacobianBuilder3D.hpp(126): warning C4244: '初期化中': '__int64' から 'int' への変換です。データが失われる可能性があります。
C:\Users\kouno\Desktop\Projects\MonadStudio\src\ssj\JacobianBuilder3D.hpp(146): warning C4244: '初期化中': '__int64' から 'int' への変換です。データが失われる可能性があります。
C:\Users\kouno\Desktop\Projects\MonadStudio\src\ssj\JacobianBuilder3D.hpp(167): warning C4244: '初期化中': '__int64' から 'int' への変換です。データが失われる可能性があります。
C:\Users\kouno\Desktop\Projects\MonadStudio\src\ssj\JacobianBuilder3D.hpp(185): warning C4244: '初期化中': '__int64' から 'int' への変換です。データが失われる可能性があります。
C:\Users\kouno\Desktop\Projects\MonadStudio\src\ssj\FakeNewsAggregator.hpp(132): warning C4244: '=': '__int64' から 'int' への変換です。データが失われる可能性があります。
C:\Users\kouno\Desktop\Projects\MonadStudio\src\ssj/GeneralEquilibrium.hpp(147): warning C4267: '初期化中': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
C:\Users\kouno\Desktop\Projects\MonadStudio\src\InequalityAnalyzer.hpp(56): warning C4244: '初期化中': 'Eigen::EigenBase<Derived>::Index' から 'int' への変換です。データが失われる可能性があります。
        with
        [
            Derived=Eigen::Matrix<double,-1,1,0,-1,1>
        ]
src/python_bindings.cpp(52): warning C4267: '=': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(105): warning C4267: '引数': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(165): warning C4267: '引数': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(175): error C2039: 'OneAssetSolver': 'Monad' のメンバーではありません
C:\Users\kouno\Desktop\Projects\MonadStudio\src\OptimalPolicy.hpp(7): note: 'Monad' の宣言を確認してください
src/python_bindings.cpp(175): error C2065: 'OneAssetSolver': 定義されていない識別子です。
src/python_bindings.cpp(175): error C2146: 構文エラー: ';' が、識別子 'solver' の前に必要です。
src/python_bindings.cpp(175): error C3861: 'solver': 識別子が見つかりませんでした
src/python_bindings.cpp(177): error C2065: 'solver': 定義されていない識別子です。
src/python_bindings.cpp(201): error C2065: 'solver': 定義されていない識別子です。
src/python_bindings.cpp(202): error C2065: 'solver': 定義されていない識別子です。
src/python_bindings.cpp(234): warning C4244: '初期化中': '__int64' から 'int' への変換です。データが失われる可能性があります。
src/python_bindings.cpp(285): warning C4267: '引数': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(301): warning C4244: '引数': 'Eigen::EigenBase<Derived>::Index' から '_Ty' への変換です。データが失われる可能性があります。
        with
        [
            Derived=Eigen::SparseMatrix<double,0,int>
        ]
        and
        [
            _Ty=int
        ]
src/python_bindings.cpp(302): warning C4244: '引数': 'Eigen::EigenBase<Derived>::Index' から '_Ty' への変換です。データが失われる可能性があります。
        with
        [
            Derived=Eigen::SparseMatrix<double,0,int>
        ]
        and
        [
            _Ty=int
        ]
src/python_bindings.cpp(324): warning C4267: '引数': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(366): warning C4244: '初期化中': 'Eigen::EigenBase<Derived>::Index' から 'int' への変換です。データが失われる可能性があります。
        with
        [
            Derived=Eigen::Matrix<double,-1,1,0,-1,1>
        ]
src/python_bindings.cpp(408): warning C4267: '初期化中': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(461): warning C4267: '引数': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(489): warning C4267: '初期化中': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(538): warning C4267: '引数': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(545): warning C4267: '初期化中': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(577): warning C4267: '引数': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(584): warning C4267: '初期化中': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(615): warning C4267: '引数': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(622): warning C4267: '初期化中': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
src/python_bindings.cpp(649): warning C4267: '引数': 'size_t' から 'int' に変換しました。データが失われているかもしれません。
error: command 'C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Tools\\MSVC\\14.50.35717\\bin\\HostX86\\x64\\cl.exe' failed with exit code 2
