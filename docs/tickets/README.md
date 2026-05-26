# チケット運用ルール

`docs/milestones.md` の各サブタスクに対して 1 枚ずつチケットを切り、実装〜レビューまでをトレースする。

## ファイル命名規則

- `T-<milestone>.<index>-<short-kebab>.md` 形式
- 例: `T-M0.1-python-env.md`, `T-M1.4-generator.md`
- マイルストーン番号 / サブタスク番号は `docs/milestones.md` の見出しと完全一致させる

## ステータス遷移

```
pending ─► in_progress ─► in_review ─► completed
                          │
                          └─► blocked  (依存が解消されたら in_progress に戻す)
```

各チケット先頭の YAML フロントマターで `status:` を更新。`docs/tickets/index.md` も同時に更新する。

## チケットに含めるセクション (テンプレート参照)

`_TEMPLATE.md` の構造を厳守する。

1. タスク目的とゴール
2. 実装内容の詳細
3. エージェントチームの役割と人数
4. 提供範囲 (Scope / Out of Scope / Deliverable)
5. テスト項目 (Unit / e2e / Acceptance criteria)
6. 懸念事項
7. レビュー観点
8. ゼロから作り直すとしたら (フェーズ完了後にエージェントチームで再評価)
9. 後続タスクへの連絡事項

## マイルストーンとの相互リンク

- 各チケット: フロントマター `milestone:` でマイルストーンを参照、本文冒頭で `docs/milestones.md` の該当節へリンク
- `docs/milestones.md`: 各サブタスク見出し直下に `**チケット**: [T-MX.Y](tickets/T-MX.Y-<short>.md)` を追記
- `docs/tickets/index.md`: 全チケット一覧と進捗を集約

## コミットルール

- **1 チケット作成 = 1 コミット**
- コミットメッセージ: `tickets(MX): add T-MX.Y <short-title>`
- フェーズ (マイルストーン) 完了時の「ゼロから作り直すとしたら」一括レビュー後の修正は、`tickets(MX): refine "rebuild from scratch" sections` でまとめてコミット

## フェーズ完了時のレビュー

1 マイルストーン分のチケットがすべて in_review/completed になった時点で、エージェントチームを起動して当該マイルストーンの **全チケットの §8 「ゼロから作り直すとしたら」セクション** を相互参照しながら見直す。

- 観点: フェーズ全体の整合性、設計上の見落とし、より良い分解の余地
- アウトプット: 各チケット §8 の更新、必要なら §6 懸念事項にも反映
- レビュー完了後に次マイルストーンへ進む

## 進捗の見方

- 全体進捗: `docs/tickets/index.md` の "Status" 列
- マイルストーン進捗: `docs/milestones.md` の各節
- フェーズレビュー結果: 各チケット §8 の更新履歴 (フロントマター `updated:` で追跡)
