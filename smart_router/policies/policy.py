import logging 
from abc import ABC, abstractmethod
from typing import List, Optional
from smart_router.worker import Worker
from smart_router.config.policy import PolicyConfig


class Policy(ABC):

    @abstractmethod
    def name(self) -> str:
        return "round_robin"

    @abstractmethod
    def select_worker(self, workers: List[Worker], *args, **kwargs) -> Optional[Worker]:
        """Select a worker from the list of available workers."""
        raise NotImplementedError("select_worker method must be implemented by subclasses")
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of the policy."""
        raise NotImplementedError("name method must be implemented by subclasses")
    

def get_policy_config(config: PolicyConfig) -> Policy:
    """Factory method to get a policy instance by name."""
    policy_name = config.policy
    if policy_name == "round_robin":
        from smart_router.policies.round_robin import RoundRobinPolicy
        return RoundRobinPolicy(config)
    
    elif policy_name == "minimum_load":
        from smart_router.policies.minimum_load import MinimumLoadPolicy
        return MinimumLoadPolicy(config)
    
    elif policy_name == "consistent_hash":
        from smart_router.policies.consistent_hash import ConsistentHashPolicy
        return ConsistentHashPolicy(config)
    
    elif policy_name == "power_of_two":
        from smart_router.policies.power_of_two import PowerOfTwoPolicy
        return PowerOfTwoPolicy(config)
    
    elif policy_name == "prefix_aware":
        from smart_router.policies.prefix_aware import PrefixAwarePolicy
        return PrefixAwarePolicy(config)
    elif policy_name == "kv_aware":
        from smart_router.policies.kv_aware import KvAwarePolicy
        return KvAwarePolicy(config)
    else:
        raise ValueError(f"Unknown policy name: {policy_name}")
