# CurlingCNN: CNN based curling AI

## Overview
CurlingCNN is a CNN based curling AI that works on DigitalCurling4.
- [Website](http://minerva.cs.uec.ac.jp/cgi-bin/curling/wiki.cgi?page=FrontPage)
- [GitHub](https://github.com/digitalcurling/DigitalCurling)

The CNN takes an image of sheet as input, and predicts the winning probability and the next shot.
The CNN is trained with the game statistics from World Curling Championships and Olympics.
This program is inspired by AlphaGo Zero and dlshogi.

## 必要要件

#### ツール
- C++17 対応のコンパイラ (GCC, Clang, MSVC, etc.)
- Git
- [CMake](https://cmake.org/) 3.26 以上
- [OpenSSL](https://www.openssl.org/) (HTTPS通信を行う場合)

#### 依存ライブラリ
- LibTorch
- CUDA
- CuDNN
These required libraries are automatically resolved by [Dockerfile]{./Dockerfile}.

以下のライブラリは CMake によって自動的に解決されます。
- [DigitalCurling](https://github.com/digitalcurling/DigitalCurling) 4.0 以上
- [CLI11](https://github.com/CLIUtils/CLI11)
- [httplib](https://github.com/yhirose/cpp-httplib)

> [!Important]
> HTTPS通信を行う場合、OpenSSLが必要です。
> OpenSSLはシステムにインストールされている必要があります。

## ビルド

1. ビルド用ディレクトリを作成し、ビルドを実行します。
   ```bash
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build . --config Release
   ```

## 使用方法

ビルド成果物を実行することで、思考エンジンクライアントが起動します。
以下の実行引数が使用可能です。

#### フラグ
| 引数 | 説明 |
|------|------|
| `--help`, `-h` | ヘルプメッセージを表示します。 |
| `--version`, `-v` | バージョン情報を表示します。 |
| `--debug`, `-d` | デバッグモードを有効にして起動します。 |
| `--console`, `-c` | コンソール入力を有効にして起動します。 |

#### オプション
| 引数 | 説明 | デフォルト値 |
|------|------|--------------|
| `--host` | 接続先ホストを指定します。 | none |
| `--id` | 接続先のゲームIDを指定します。 | none |
| `--team` | 接続するチームを指定します。(0または1) | 0 |
| `--auth-id` | Basic認証のIDを指定します。 | `user` |
| `--auth-password` | Basic認証のパスワードを指定します。 | `password` |
| `--plugin-dir` | プラグインのディレクトリを指定します。 | `./plugins` |

オプションは全て任意オプションですが、`--host` および `--id` はクライアントの起動に必要です。  
`--console` フラグ指定を指定した場合は、標準入力にて接続先情報を入力することができます。

## 思考エンジンの開発方法

思考エンジンは、[src/engine/](src/engine/) ディレクトリ内に配置。

#### 開発手順

1. 思考エンジンのクラスを作成します。  
対応させるルールに応じて、以下のインターフェイスクラスを継承してください。
複数のクラスを継承し、複数のルールに対応する思考エンジンを作成することも可能です。
   - `IStandardThinkingEngine` (通常のカーリング用)
   - `IMixedThinkingEngine` (ミックスカーリング用)
   - `IMixedDoublesThinkingEngine` (ミックスダブルスカーリング用)

1. `src/client_setup.cpp` 内の関数を編集し、作成した思考エンジンのクラスを返すようにします。  
その他のコードは、必要に応じて編集してください。

1. ルートディレクトリの `client_config.cmake` でクライアント情報を設定します。

## Model
The network takes the sheet, score difference, and shot number as inputs.
Policy network selects the next shot (speed, angle, and rotation).
Value network predict the score of the end, which is converted to the winning rate with the number of ends to be played and the score difference.

## Training
Game statistics extracted from shot by shot available on [curlit](https://curlit.com/results) are used to train the model. Data extraction is described [here](https://www.jordanmyslik.com/portfolio/curling-analytics/).

## Strategy
CurlingCNN estimates shot candidates (speed, angle, and rotation) from the current state.
Then, it adopts a shot where the model returns the highest winning probability after simulation of shot candidates.
This strategy is inspired by AlphaGo Zero and dlshogi.

## Author
Takehiro Yoshioka
- [@pon_pokapoka](https://x.com/pon_pokapoka)

## ライセンス

[MIT](./LICENSE)
