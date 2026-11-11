# Known Issues

> 更新：2026-11-10。条目 3、5 已随新实验更新；新增 7–9（多变量观测与训练诊断中发现的坑）。

1. **Dataset cold init ≈ 3 min.** `SingleCycleDADataset.__init__` scans all hourly
   partitions to enumerate stations (~124 files Jan window; disk-cached afterwards).
   Mitigation planned: persist per-month consolidated observation arrays + station lists.

2. **gcsfs 2026.10.1 path bug for non-zarr objects.** `fs.open("gs://…")` mangles object
   paths; zarr reads unaffected. Workaround: METAR parquet fetched via plain HTTPS
   (`utils/io.py`). Revisit if gcsfs is fixed.

3. **METAR DD flags B/G/Q/X are not officially documented** in the WB2 export.
   ~[已部分处理] 五变量接入(single_da_06)时按变量重查并沿用接受策略；
   扩展到新变量/时段时仍需复查。

4. **`timeReceived` slightly before `timeObs`** for ~3% of reports (negative lag,
   typically ≤ 2 min, stations reporting early). Current cutoff only enforces
   `timeReceived ≤ t+30min`, so these pass. No evidence of harm; documented in
   DATA_AUDIT §4.

5. **Representativeness floor — 已量化（obs_diagnostics_2021，2021 全年）**：
   站点创新量方差 **94.8% 来自 d=y−H(ERA5)**（RMS 2.34K），信号 r 仅占 5.2%；
   逐循环 corr(创新量, r)=0.211。高程 >1500m 站 d=4.0K（近纯噪声）。
   站点空间任何"同化收益"声明必须对照此底噪分层报告（现行做法有效）。

6. **Pangu zarr lacks units attributes**; units follow WB2 conventions (K, m s-1, Pa).

7. **多变量观测 ID 映射易错**：`metar_multivar_cache` 中 var 3=露点、var 4=MSLP
   （与直觉相反；SPECS 为准）。step2 基线曾因此把露点当 MSLP 订正——消费观测时
   一律以 `offgrid_da.data.metar_multivar.SPECS` 为映射来源，勿手写。

8. **MSLP 高海拔站归算长尾**：局地订正实验中，真实 MSLP 创新量含高海拔站点
   海平面归算差异的长尾，最优增益因此收敛到 0（step2_local_baseline）。
   使用真实 MSLP 前应考虑按高程过滤/降权。

9. **训练未收敛警示**：single_da_09 与 step2 合成臂的 best_step 均为预算末步
   (43840)。配对结论不受影响，绝对数字解读需注明"未饱和"。
   [已跟进 2026-11-10/11] 双臂匹配延长 5 epochs（single_da_10）：best_step
   又停在新末步——系 **cosine LR 尾部效应**（末端每步微改善使 best 被动
   跟随），非预算不足；延长只收回与局地基线缺口的 5%（real）/9.5%（syn）。

   Verified for T2M only in this round; re-verify for wind variables in Phase 5.

7. **ERA5 zarr time coverage ends 2023-01-10** — fine for the 2018–2022 protocol, but
   blocks extending test years beyond 2022 without switching datasets (would be a
   protocol change requiring plan update).

8. **Station split over Jan-2020 window only.** The full 2018–2020 station pool (stations
   appearing/disappearing over years) must be rebuilt for full training; current split is
   minimal-window-scoped by design.

9. **`_read_hour` LRU cache is per-process**; multi-worker DataLoader (num_workers>0)
   re-downloads from local disk cache but re-parses parquet per worker. Acceptable for
   M2 single-GPU training; profile before scaling.
