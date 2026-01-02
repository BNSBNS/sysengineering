"""Streaming system value objects."""

from typing import NewType

# Type-safe identifiers
NodeId = NewType('NodeId', str)
TopicName = NewType('TopicName', str)
PartitionId = NewType('PartitionId', int)
ConsumerId = NewType('ConsumerId', str)
ConsumerGroupId = NewType('ConsumerGroupId', str)
Offset = NewType('Offset', int)
Term = NewType('Term', int)

# Record identifiers
RecordKey = NewType('RecordKey', bytes)
RecordValue = NewType('RecordValue', bytes)


def create_node_id(broker_name: str) -> NodeId:
    """Create a node ID.
    
    Args:
        broker_name: Broker hostname/address.
        
    Returns:
        Node ID.
    """
    return NodeId(f"node-{broker_name}")


def create_consumer_id(group: str, instance: int) -> ConsumerId:
    """Create a consumer ID.
    
    Args:
        group: Consumer group name.
        instance: Instance number.
        
    Returns:
        Consumer ID.
    """
    return ConsumerId(f"{group}-consumer-{instance}")


def create_consumer_group_id(name: str) -> ConsumerGroupId:
    """Create a consumer group ID.
    
    Args:
        name: Group name.
        
    Returns:
        Consumer group ID.
    """
    return ConsumerGroupId(f"group-{name}")
