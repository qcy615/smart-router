# -*- coding: utf-8 -*-

# Re-export all public types and functions for convenient access
from smart_router.policies.policy import Policy, get_policy_config
from smart_router.policies.round_robin import RoundRobinPolicy
from smart_router.policies.consistent_hash import ConsistentHashPolicy
from smart_router.policies.prefix_aware import PrefixAwarePolicy
from smart_router.policies.kv_event_prefix_aware import KVEventPrefixAwarePolicy
from smart_router.policies.power_of_two import PowerOfTwoPolicy

__all__ = [
    "Policy",
    "get_policy_config",
    "RoundRobinPolicy",
    "ConsistentHashPolicy",
    "PrefixAwarePolicy",
    "KVEventPrefixAwarePolicy",
    "PowerOfTwoPolicy",
]
