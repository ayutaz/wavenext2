---
id: T-MX.Y
title: <短いタイトル>
milestone: MX
phase: MX
status: pending  # pending | in_progress | in_review | completed | blocked
size: small | medium | large
owner: -
created: 2026-MM-DD
updated: 2026-MM-DD
depends_on: []          # 先行 T-... を列挙
blocks: []              # 後続 T-... を列挙
related_docs:
  - docs/milestones.md#mxy
  - docs/architecture.md
  - docs/training.md
  - docs/implementation-plan.md
  - docs/open-questions.md
---

# T-MX.Y: <短いタイトル>

> **マイルストーン**: [MX](../milestones.md#mx) / **サブタスク**: [MX.Y](../milestones.md#mxy)
> **依存**: <T-... のリンク> / **後続**: <T-... のリンク>

## 1. タスク目的とゴール

### 目的
このタスクで解決する課題は何か (1〜2 文)。

### ゴール
完了したと判断できる具体的な状態 (3〜5 項目)。
- [ ] ...
- [ ] ...

## 2. 実装内容の詳細

### 2.1 対象ファイル
- 新規: `src/.../foo.py`, `tests/test_foo.py`
- 編集: -

### 2.2 主要構造
```python
# 主要なクラス / 関数のシグネチャを抜粋
class Foo(nn.Module):
    def __init__(self, ...): ...
    def forward(self, x): ...
```

### 2.3 使用するハイパーパラメータ / 定数
| 名前 | 値 | 出典 |
|---|---|---|
| ... | ... | docs/architecture.md §X |

### 2.4 アルゴリズム / 処理フロー
1. ...
2. ...

## 3. エージェントチームの役割と人数

| 役割 | 人数 | 担当範囲 | 推奨 subagent_type |
|---|---|---|---|
| Implementer | 1 | 実装 + Unit テスト記述 | general-purpose |
| Reviewer | 1 | コードレビュー + 論文整合性確認 | general-purpose |
| Tester | 1 | テスト実行 + Acceptance 検証 | Explore |

### 並列度
- 同フェーズ内の他チケットと並列実行可能か: yes / no (依存関係に基づく)
- 並列実行する場合の最大並列数: N

## 4. 提供範囲 (Scope)

### In Scope
- 実装する機能・モジュール
- 書くテスト

### Out of Scope
- このチケットでは扱わないもの (次チケット送り or 別マイルストーン)

### Deliverable
- ファイル: ...
- 関数 / クラス: ...
- ドキュメント差分: ...

## 5. テスト項目

### 5.1 Unit テスト
- [ ] `tests/test_<module>.py::test_shape` — 入出力 shape が仕様通り
- [ ] `tests/test_<module>.py::test_param_count` — パラメータ数が論文 Table と一致
- [ ] `tests/test_<module>.py::test_gradient_flow` — backward で全パラメータに grad
- [ ] `tests/test_<module>.py::test_deterministic` — 同入力 / 同 seed で同出力

### 5.2 e2e / 結合テスト
- [ ] 上位モジュール (`sub_model.py` / `gan_wavenext2.py` 等) からの呼び出しが成功
- [ ] 実音声 1 utterance を通して forward + backward が成功

### 5.3 Acceptance criteria (`docs/milestones.md` §MX.Y より転記)
- [ ] ...
- [ ] ...

## 6. 懸念事項

### 6.1 技術的リスク
- リスク内容、影響範囲、検知方法

### 6.2 仕様の曖昧さ
- `docs/open-questions.md` の §... を参照、未解決ならここで決定

### 6.3 他チケットとの整合性
- T-... と仕様が食い違う可能性、すり合わせ要否

## 7. レビュー観点

実装完了後、Reviewer が以下を確認:

- [ ] 論文記述 / `docs/architecture.md` との整合性
- [ ] Acceptance criteria 全項目クリア
- [ ] Unit テスト全 pass
- [ ] CLAUDE.md / 既存コードのスタイル準拠 (型ヒント、docstring、命名)
- [ ] エラー処理: 不正入力時の挙動 (assert / raise)
- [ ] パラメータ数 / メモリ消費が想定内
- [ ] 参考実装をコピーしていない (一から書いたか)

## 8. ゼロから作り直すとしたら

> このセクションはチケット作成時に初稿を書き、**フェーズ (マイルストーン) 完了時にエージェントチームで再評価して必要なら更新する**。

### 8.1 別の設計を採るとしたら
- 採用可能だった代替設計、なぜ採用しなかったか
- 代替案を採った場合のメリット / デメリット

### 8.2 思想 / 哲学の見直し
- このサブタスクの粒度は適切か
- 別マイルストーンに移すべき部分はないか
- インターフェース定義の見直し余地

### 8.3 学んだこと (チケット完了後に追記)
- 実装中に判明した想定外
- 次の似たタスクで応用できる教訓

## 9. 後続タスクへの連絡事項

### 9.1 後続チケットに渡す情報
- インターフェース: `class Foo` の signature、`forward()` の I/O 仕様
- 設定値: YAML config に追加したキー
- 注意事項: 後続が踏みそうな罠

### 9.2 ドキュメント更新
- 完了時に更新するドキュメント:
  - [ ] `docs/milestones.md` の Acceptance チェックボックス
  - [ ] `docs/tickets/index.md` のステータス
  - [ ] (該当時) `docs/architecture.md` / `docs/training.md` の差分

### 9.3 Open question として残ったもの
- 解決できなかった疑問、`docs/open-questions.md` への追記要否
