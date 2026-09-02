# Legacy root pipeline

整理前的 Factor/Teacher 候选构造阶段当前仍以兼容入口保留在 `code/` 根目录，
因为仓库旧测试和部分复现实验按原文件名动态加载它们。当前 Top-8 简化 Trace
两阶段流程不会调用这些文件。

共享 Ranker 的底层实现 `07_build_ranker_data.py` 和
`08_train_global_ranker.py` 仍保留在根目录，因为当前包装器会复用它们。
