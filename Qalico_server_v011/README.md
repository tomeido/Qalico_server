Qalico server
====

# Overview
![Qqlico](https://drive.google.com/uc?export=view&id=1FUD4A2OQv8z7u_ckAh4D2h3XmbGxYJ91)
Grasshopper用平面計画支援プラグイン"Qalico"を動作させるためのサーバープログラムです。プラグイン本体は下記サイトより取得してください。  
- [food4Rhino "Qalico"](https://www.food4rhino.com/en/app/qalico)  

Qalicoは量子アニーリング技術を使って多目的最適化問題を解き、ゾーニングや通路計画の試行錯誤を高速化します。量子アニーリングマシン(QAM)にはFixstarsの[Amplify](https://amplify.fixstars.com/ja/)を使っており、Qalico serverはQalicoプラグインとAmplifyのブリッジ処理を行います。

# Description
Qalico serverは、Grasshopper上で動作するQalicoプラグインと通信し、QAMで処理を行うためのパラメーターを生成してQAMにデータを送り、計算結果をQalicoプラグインに戻します。実装にはFlaskを使用しpythonで記述しています。

# Demo
Qalicoプラグインを使って平面計画をしている様子です。この裏側でQalico serverとQAMが動作しています（別ウィンドウで動画再生）。

[![](https://img.youtube.com/vi/SRauzsvLN2Y/0.jpg)](https://www.youtube.com/watch?v=SRauzsvLN2Y)

# Requirement
Qalico serverの動作にはpython3の処理系が必要です。  
WSLにて動作確認済ですが、windows、MacOS、Lunix系でも動作すると思います。  
なお、Qalicoプラグインの動作には、Rhinoceros7以上が必要です。詳細は[food4RhinoのQalicoページ](https://www.food4rhino.com/en/appa/qalico)を参照ください。  
また、QAMのFixstar Amplifyを使うためには、事前にアクセストークンの入手が必要です。下記サイトより各自で入手してください。

- [Amplifyアクセストークン入手](https://amplify.fixstars.com/ja/#gettingstarted)  

**Qalicoプラグインの動作確認だけであれば、別途テスト用のQalicoサーバーを提供していますので、本Qalicoサーバーをインストールしなくても使うことができます。**

# Install
1.このリポジトリを適切なフォルダにクローンします。  
2.venv等でpythonの仮想環境を構築することをお勧めします。  
3.requirement.txtを使って必要なパッケージを一式インストールして下さい。  
4.入手したAmplifyアクセストークンを環境変数"AMPLIFY_API_TOKEN"にセットしてください。  

コマンドの例( XXXXXX の部分は入手したトークンを入力)
```
> git clone https://github.com/TagawaN/Qalico_server.git ./Qalico_server
> cd Qalico_server
> python3 -m venv .venv
> source .venv/bin/activate
> (env)pip install -r requirements.txt
> (env)export AMPLIFY_API_TOKEN="XXXXXX"
```

# Usage
ローカルマシンでQalicoサーバーを動作する場合  
1.Qalicoサーバーを起動
```
> flask run
```
http://localhost:5000 でサーバーが起動し、Qalicoプラグインとの通信待ち状態となります。  
2.QalicoプラグインにQalicoサーバーのURLを設定する  

- zooning_opt module: http://localhost:5000/zoning_opt  
- corridor_layout_opt module: http://localhost:5000/corridor_layout_opt  

3.Grasshopper上でQalicoプラグインのパラメーター設定し、Activeにする。  


# Acknowledgements
本ソフトウエア開発は、[2021年度IPA未踏ターゲットプロジェクト](https://www.ipa.go.jp/jinzai/target/2021/gaiyou_ts-1.html)の支援を受け、「建築パラメトリックデザインのための対話的動線計画最適化」の取り組みの成果です。

# Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

# Author
田川直樹(tagawa.naoki@teragroove.com)  
Emiri Yoshinari(artwork)
