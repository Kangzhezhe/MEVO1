# 实验产物命名规范

`logs/` 和 `result/` 在项目中是符号链接，真实数据位于：

- `/data/liux/MEVO_global_cot/logs`
- `/data/liux/MEVO_global_cot/result`

所有正式实验使用统一 ID：

```text
YYYYMMDD_HHMMSS_experiment_name
```

结果目录结构：

```text
result/YYYYMMDD_HHMMSS_experiment_name/
├── editor/
│   ├── checkpoints/
│   ├── final_adapter/
│   └── global_idpo/
├── predictions/
└── reports/
```

日志文件结构：

```text
logs/YYYYMMDD_HHMMSS_experiment_name.log
```

启动新实验时使用：

```bash
cd /home/liux/kk/MEVO_global_cot
code/run_timestamped_global.sh config_global.yaml
```

脚本会冻结一份本次运行配置到
`result/YYYYMMDD_HHMMSS_experiment_name/run_config.yaml`，并让日志、checkpoint、
最终 adapter、预测和报告共享同一个实验 ID。正式流水线会拒绝未带时间戳的结果
路径，因此不要直接复用固定结果目录运行正式实验。
