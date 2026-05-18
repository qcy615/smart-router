# Kubernetes Service Discovery

When `--enable-k8s-discovery` is set, `smart_router` does not require
static worker URL arguments. In PD disaggregation mode this replaces
`--prefill-urls` and `--decode-urls`; in normal mode this replaces
`--worker-urls`. The engine process watches Kubernetes pods in the same
inference task and dynamically registers/removes workers.

## Pod Requirements

- The router pod and all worker pods must share the same task label, default
  `task_id=<value>`.
- Worker pods must set `WORKERTYPE=PREFILL`, `WORKERTYPE=DECODE`, or
  `WORKERTYPE=REGULAR`.
- Use `PREFILL` and `DECODE` workers with `--pd-disaggregation`; use `REGULAR`
  workers in normal non-PD mode.
- Pods with `HEADLESS=true` are ignored. Use this for distributed worker pods
  that do not expose the inference HTTP endpoint.
- Only `Running` pods with a Pod IP are registered.

## Worker URLs

Worker URLs are built from Pod IP and the configured port:

- `PREFILL`: `http://<podIP>:<prefill-port>`
- `DECODE`: `http://<podIP>:<decode-port>`
- `REGULAR`: `http://<podIP>:<regular-port>`

## Useful Options

PD disaggregation mode:

```bash
--enable-k8s-discovery
--pd-disaggregation
--k8s-prefill-port 8100
--k8s-decode-port 8200
--k8s-task-label-key task_id
--k8s-namespace inference
```

Normal mode:

```bash
--enable-k8s-discovery
--k8s-regular-port 8300
--k8s-task-label-key task_id
--k8s-namespace inference
```

When K8S discovery is enabled, PD mode requires `--k8s-prefill-port` and
`--k8s-decode-port`; normal mode requires `--k8s-regular-port`.

If `--k8s-namespace` is not provided, the router reads the namespace from the
mounted service account. The Kubernetes Python SDK uses in-cluster config by
default and falls back to local kubeconfig when running outside a cluster.

## RBAC

The router service account needs permission to read and watch pods:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: smart-router-discovery
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: smart-router-discovery
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: smart-router-discovery
subjects:
  - kind: ServiceAccount
    name: smart-router
```

## Scheduling Behavior

Newly discovered workers start as unhealthy and are added to scheduling only
after their `/health` endpoint returns HTTP `200`. A debounced health refresh is
triggered when new workers are discovered, so workers can become schedulable soon
after they are ready without waiting for the next full health-check interval.
Removed or non-ready pods are removed from future scheduling. In-flight requests
keep using the worker URL selected at scheduling time.
