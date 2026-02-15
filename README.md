# Spectrum Package

## インストール

### 前提条件

- Python 3.8以上
- [Poetry](https://python-poetry.org/)

### Windows用インストールガイド

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
