Smart Router

A high-performance, production-grade request router for LLM inference serving. Supports Prefill-Decode (PD) disaggregation, prefix-aware KV-cache routing, native Kubernetes (k8s) service discovery, and integration with multiple inference backends (vLLM, SGLang).

![arch](./imgs/arch.png)

Key Features

- Core Architecture: Request routing framework and async processing patterns
- Load Balancing: Multiple algorithms (prefix aware, power of two, consistent hashing, minimum load, round robin)
- Prefill-Decode Disaggregation: Specialized routing for separated processing phases
- Service Discovery: Kubernetes-native worker management and health monitoring

- Multi-Backend Support — vLLM and SGLang inference engines
- Data-Parallel Awareness — Support for intra-node data-parallel worker groups
- Built-in Benchmark — Multi-turn benchmarking tool for evaluating routing performance

Installation

    #Install from source
    pip install .
    
    #Install with benchmark dependencies
    pip install .[benchmark]
    

Or use uv:

    uv sync
    uv sync --extra benchmark  # with benchmark dependencies



Docker

    docker build -t smart-router .
    
    # With benchmark extras
    docker build --build-arg INSTALL_BENCHMARK=true -t smart-router .

Quick Start

Regular HTTP Routing

    python -m smart_router serve \
    	--router-type vllm \
    	--policy power_of_two \
    	--worker-urls http://worker1:8000 http://worker2:8000 \
    	--worker-intra-dp-size 4

Prefill/Decode Disaggregation (PD)

    python -m smart_router serve \
        --router-type vllm \
        --pd-disaggregation \
        --prefill-urls http://worker1:8000 \
        --decode-urls http://worker2:8000 \
        --prefill-policy power_of_two \
        --decode-policy power_of_two \
        --prefill-intra-dp-size 2 \
        --decode-intra-dp-size 2

## Documentation

- [Kubernetes Service Discovery](./docs/kubernetes-service-discovery.md)
- [Prefix-Aware Tree Eviction](./docs/prefix-aware-tree-eviction.md)

## benchmark
Run the integrated benchmark entrypoint:

    python -m smart_router benchmark --input-file conversations.json --model /path/to/model --url http://127.0.0.1:8000

RoadMap

- SGLang support
- Service discovery
- vllm kv event report
- batch schedule
- prompt bin packing policy
