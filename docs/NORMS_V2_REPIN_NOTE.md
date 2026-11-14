# arches_train_2018_2020.json 版本记录（2026-11-15）

## 现状
- 当前文件 = **v2-regen**（迁移重建期间由 prepare_arches_norms.py 从再生成背景重算）
  sha256 = `d06d65895dc366eac7a94a0c75e232311b0a1aa7a279f814fd2af9c565dd6e27`
- 原版（v1）sha256 = `dd6bbd47ce91dad62af666b3af6b8d292ba540196e4778afad3c4b6ff05d7f6d`（由 single_da_05/normalization.json 的守卫钉住）
- v1 字节在本机**不可恢复**：迁移清掉了原始背景（其统计只能由原始数据逐步累积复现，
  幸存 norm_stats 副本数值与 v1 差在浮点尾数，字节重组对 sha 验证未命中）。
  **v1 全球仅存于迁移目标机** `outputs/normalization/arches_train_2018_2020.json`；
  若日后可达：scp 拉回 → sha 对上 `dd6bbd47ce91…` → 直接归位本文件并回退下述重钉。

## v1 vs v2 数值差异（v1 以幸存 norm_stats_train_2018_2020.json 为代理，同 4383 周期口径）
| 量 | v1(代理) | v2(regen) | 差 |
|---|---|---|---|
| background mean | 278.864867 | 278.848388 | -0.016479 K |
| background std | 21.140688 | 21.194487 | +0.254% |
| reference mean | 278.850729 | 278.850361 | -0.000367 K |
| reference std | 21.185890 | 21.185472 | -0.002% |
| n（网格点）| 127282320 (4383 周期) | 127311360 (4384 周期) | +1 周期 |

## 根因链
1. 迁移丢弃原始 geoarches/modelstore 背景与 v1 json 所依赖的原始 arches_backgrounds；
   再生成背景固有 ~0.25% std 漂移（GPU 非确定性 + 4383→4384 周期差）。
2. 重建 norms 链时 v1 json 被覆盖（它在 rsync 白名单内本应幸存——操作失误，已披露）。
3. da_sample_cache（5844 .pt）与 metar protocol 均以 v2 统计构建，内部自洽；
   预训练 DA 模型复评因此携带 ~0.05K 量级背景/归一化漂移，收官报告须如实记录。

## 守卫处理
- `outputs/single_da_05_input82_ablation/normalization.json` 的
  `legacy_t2m_normalization_sha256` 由 v1 重钉为 v2（`d06d65895dc366ea…`）。
  该守卫用途是防静默漂移；本次漂移非静默（本文件 + git 记录）。
