# Spectrum Package

## 注意

このパッケージは開発途中です。そのため、パッケージの構成やAPIが変更される可能性があります。
また、最終目標となる`HST/STIS`の分光データから3次元データキューブ（スリット垂直方向×スリット長方向×波長方向）を作成する段階まで到達しておりません。
以下に現在の実装段階を記載します。

### プロジェクトの進捗とデータフロー

本プロジェクトでは、以下のデータフローに基づいて実装を進めています。
現在までの実装状況は以下の通りです。

- [x] **calibrated file**
  - `InstrumentModel`, `STISFitsReader` (実装済み)
- [x] **Image data**
  - `ImageModel` (実装済み)
- [ ] **continuum-subtracted data**
  - (未実装 - 今後 `ImageModel` 等の単一スリットクラスに統合予定)
- [ ] **removed data**
  - (未実装 - 宇宙線除去・エラー除去処理)
- [ ] **extracted data**
  - `VelocityModel` (部分的に実装済みだが、Image data との統合クラスへ移行予定)
- [ ] **3D data Cube**
  - `VelocityMap` (部分的に実装済みだが、全スリットをまとめる3Dクラスへ移行予定)

> **設計方針**:
> `Image data` から `extracted data` までの処理を一元管理する単一スリット用のクラスを作成し、それら（6本のスリット）を束ねて `3D data Cube` を扱うクラスとして実装します。

### 今後改修・実装予定のクラス

- **単一スリット管理クラス (New/Refactor)**
    - 現行の `ImageModel` と `VelocityModel` を統合・拡張。
    - Calibrated file から Extracted data までのパイプライン（連続光除去、データ除去含む）を担当。
- **3Dデータキューブクラス (New/Refactor)**
    - 現行の `ImageCollection` や `VelocityMap` を統合・拡張。
    - 複数の単一スリットオブジェクトを保持し、3次元的な構造解析や可視化を担当。

## インストール

### 前提条件

- Python 3.8以上
- [Poetry](https://python-poetry.org/)

## Windows用インストールガイド

Windowsで `pip` を使ってPoetryをインストールした後、`poetry` コマンドが認識されない場合は、以下の手順に従ってください：

1.  **pipを使ってPoetryをインストール:**
    ```powershell
    pip install poetry
    ```

2.  **PATHにPoetryを追加:**
    インストールディレクトリがシステムのPATH環境変数に含まれていない場合があります。
    
    PowerShellで以下のコマンドを実行して、PATHに追加してください。
    **注意:** 以下のパスに含まれる `<ユーザー名>` の部分や、Pythonのバージョン部分はご自身の環境に合わせて変更してください。

    ```powershell
    # 設定例（Microsoft Store版Python 3.13の場合）
    $targetPath = "C:\Users\<ユーザー名>\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_xxxxxxxx\LocalCache\local-packages\Python313\Scripts"
    
    $currentPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    if (-not ($currentPath.Split(';') -contains $targetPath)) {
        [System.Environment]::SetEnvironmentVariable("Path", $currentPath + ";" + $targetPath, "User")
        echo "PATHに追加しました"
    }
    ```

3.  **ターミナルの再起動:**
    変更を適用するために、ターミナル（またはVS Code）を一度閉じてから再起動してください。

4.  **インストールの確認:**
    ```powershell
    poetry --version
    ```
