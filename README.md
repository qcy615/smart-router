Smart Router

![arch](./imgs/arch.png)

## Usage

Start the router service:

```bash
python -m smart_router serve --prefill-urls http://127.0.0.1:8100 --decode-urls http://127.0.0.1:8200
```

Start with Kubernetes pod discovery:

```bash
python -m smart_router serve \
  --enable-k8s-discovery \
  --prefill-port 8100 \
  --decode-port 8200 \
  --prefill-intra-dp-size 1 \
  --decode-intra-dp-size 1
```

Router and worker pods must share the same `task_id` label. Worker pods set
`WORKERTYPE=PREFILL` or `WORKERTYPE=DECODE`; pods with `HEADLESS=true` are not
registered as worker endpoints.

Run the integrated benchmark entrypoint:

```bash
python -m smart_router benchmark --input-file conversations.json --model /path/to/model --url http://127.0.0.1:8000
```

# RoadMap

- SGLang support
- Service discovery
- vllm kv event report 
- batch schedule
- prompt bin packing policy
