# LibriTTS-R データセットの帰属表示 (Attribution)

本プロジェクトの評価・学習では **LibriTTS-R** コーパスを使用します。LibriTTS-R は
**Creative Commons Attribution 4.0 International (CC BY 4.0)** で配布されており、
利用・改変・再配布の際は出典の明示 (attribution) が義務付けられています。

> 本リポジトリには LibriTTS-R の音声本体は含まれません (`.gitignore` で除外)。
> 評価で合成音声 (= LibriTTS-R を入力とした改変物) を配布・公開する場合は、
> 下記のクレジットを必ず併記してください (M7 主観評価の web 配信を含む)。

## ライセンス
- **CC BY 4.0**: https://creativecommons.org/licenses/by/4.0/
- 配布元 (openslr): https://www.openslr.org/141/

## クレジット (引用)

LibriTTS-R:

> Y. Koizumi, H. Zen, S. Karita, Y. Ding, K. Yatabe, N. Morioka, M. Bacchiani,
> Y. Zhang, W. Han, and A. Bapna, "LibriTTS-R: A Restored Multi-Speaker
> Text-to-Speech Corpus," in Proc. Interspeech, 2023.

LibriTTS-R は **LibriTTS** を音質復元 (restoration) したものです:

> H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen, and
> Y. Wu, "LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech,"
> in Proc. Interspeech, 2019.

LibriTTS は **LibriSpeech** (CC BY 4.0) に由来します:

> V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, "LibriSpeech: An ASR
> corpus based on public domain audio books," in Proc. ICASSP, 2015.

## 改変点の明示 (CC BY 4.0 が要求)
本プロジェクトは LibriTTS-R の 24 kHz 波形を入力として neural vocoder で
**再合成** します。配布する合成音声は LibriTTS-R の改変物 (derivative) であり、
上記クレジットと CC BY 4.0 ライセンスのもとで提供されます。
