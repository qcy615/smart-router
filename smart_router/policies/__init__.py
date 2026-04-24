# -*- coding: utf-8 -*-

# Re-export all public types and functions for convenient access
from smart_router.policies.policy import Policy, PolicyRequest, get_policy_config
from smart_router.policies.round_robin import RoundRobinPolicy
from smart_router.policies.consistent_hash import ConsistentHashPolicy
from smart_router.policies.prefix_aware import PrefixAwarePolicy
from smart_router.policies.power_of_two import PowerOfTwoPolicy

__all__ = [
    "Policy",
    "PolicyRequest",
    "get_policy_config",
    "RoundRobinPolicy",
    "ConsistentHashPolicy",
    "PrefixAwarePolicy",
    "PowerOfTwoPolicy",
]
