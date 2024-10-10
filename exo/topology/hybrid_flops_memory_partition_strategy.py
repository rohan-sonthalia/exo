from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition


class HybridFLOPSMemoryPartitioningStrategy(PartitioningStrategy):
  def __init__(self, flops_weight=0.5, memory_weight=0.5):
    self.flops_weight = flops_weight
    self.memory_weight = memory_weight
  
  def partition(self, topology: Topology) -> List[Partition]:
    nodes = list(topology.all_nodes())
      
    total_flops_fp16 = sum(node[1].flops.fp16 for node in nodes)
    total_memory = sum(node[1].memory for node in nodes)
    
    partitions = []
    start = 0
    
    for node in nodes:
      # Calculate the proportion of FP16 FLOPS and memory for each node
      flops_weight = (node[1].flops.fp16 / total_flops_fp16) * self.flops_weight
      memory_weight = (node[1].memory / total_memory) * self.memory_weight
      
      # Compute the combined weight for each device based on FLOPS and memory
      combined_weight = flops_weight + memory_weight
      end = round(start + combined_weight, 5)
      
      partitions.append(Partition(node[0], start, end))
      start = end
        
    return partitions