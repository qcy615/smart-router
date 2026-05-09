# K8S Pod 服务发现方案

## Summary

为 `smart_router` 增加可选的 K8S discovery 模式：启动时不再传 `--prefill-urls/--decode-urls`，而是由 engine 进程通过 K8S API watch 同 namespace、同 `task_id` label 的 Pod，动态维护 prefill/decode worker registry。DP size 仍使用现有 `--prefill-intra-dp-size` 和 `--decode-intra-dp-size` 参数。

## Key Changes

- 新增 discovery 配置与 CLI：
  - `--enable-k8s-discovery`
  - `--prefill-port <int>`、`--decode-port <int>`，discovery 模式必填
  - `--k8s-task-label-key task_id` 默认读取 `task_id`
  - `--k8s-namespace` 可选；不传则读取 service account namespace
- discovery 与 `router_type` 解耦；`router_type` 继续只表达 `vllm-pd-disagg` 或 `sglang-pd-disagg`。
- discovery 模式下拒绝同时传入 `--prefill-urls/--decode-urls`，避免静态和动态来源混用。

## Implementation

- 新增 `smart_router/discovery/k8s.py`：
  - 使用 in-cluster service account token + `httpx.AsyncClient` 调 Kubernetes API，不引入重型 K8S client 依赖。
  - 通过 `HOSTNAME` 获取 router 自身 Pod 名，读取自身 Pod 的 `task_id` label。
  - 初始 `list pods` 后按 `resourceVersion` 建立 `watch=true` 流；断线、`410 Gone` 后重新 list 并继续 watch。
- Pod 过滤规则：
  - 必须有相同 `task_id=<router task_id>` label。
  - 必须 `status.phase=Running`、`podIP` 存在、Pod Ready。
  - 从容器 env 中读取 `WORKERTYPE=PREFILL|DECODE`。
  - 任一容器 env 中 `HEADLESS=true` 时跳过，不注册为 worker。
- URL 构造：
  - `PREFILL` 使用 `http://<podIP>:<prefill_port>`。
  - `DECODE` 使用 `http://<podIP>:<decode_port>`。
- Registry 集成：
  - 抽出 worker 注册 helper，按现有逻辑展开 DP ranks。
  - `prefill_intra_dp_size > 1` 时注册 `DPAwareWorker(base_url)@rank`。
  - `decode_intra_dp_size > 1` 同理。
  - 修正 `WorkerRegistry.register()` 为幂等/upsert，避免 Pod watch 的重复 MODIFIED 事件造成 type index 重复。
  - Pod 删除、变为 NotReady、设置 `HEADLESS=true` 或 `WORKERTYPE` 失效时，移除对应 worker。
  - 新增 worker 初始设为 unhealthy，并触发一次 debounced health refresh，只有 `/health` 返回 200 后才参与调度。
- API 辅助：
  - 增加 engine 查询当前 worker base URLs 的 request/response，用于 discovery 模式下 `/v1/models` 动态聚合 upstream models。
  - 静态 URL 模式保持现有行为不变。

## Test Plan

- CLI/config：discovery 开启时校验端口必填、URL 不可混用、DP size 为正。
- Discovery parser：覆盖 `WORKERTYPE`、`HEADLESS=true`、Ready 状态、无 PodIP、IPv6 URL。
- Watch 同步：ADDED/MODIFIED/DELETED 正确注册、更新、剔除 worker。
- Registry：重复注册不产生重复索引；删除 DP worker 会删除所有 rank。
- Engine：动态注册后健康检查通过才可调度；删除 worker 后不再被选中。
- `/v1/models`：discovery 模式从 engine 获取当前 base URLs 并聚合模型列表。

## Assumptions

- worker 推理接口使用 HTTP。
- 不带 `HEADLESS=true` 的 `PREFILL/DECODE` Pod 即视为暴露推理接口。
- router ServiceAccount 需要同 namespace Pod 的 `get/list/watch` 权限。
- discovery 只负责 worker 实例发现，不自动推导 DP size。
