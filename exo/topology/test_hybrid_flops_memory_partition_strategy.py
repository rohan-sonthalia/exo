import unittest
from exo.topology.hybrid_flops_memory_partition_strategy import HybridFLOPSMemoryPartitioningStrategy
from exo.topology.topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.partitioning_strategy import Partition


class TestRingMemoryWeightedPartitioningStrategy(unittest.TestCase):
  def test_partition(self):
    # triangle
    # node1 -> node2 -> node3 -> node1
    topology = Topology()
    topology.update_node(
        "node1",
        DeviceCapabilities(model="test1", chip="chip1", memory=2500, flops=DeviceFlops(fp32=3, fp16=5, int8=8)),
    )
    topology.update_node(
        "node2",
        DeviceCapabilities(model="test2", chip="chip2", memory=2000, flops=DeviceFlops(fp32=4, fp16=6, int8=9)),
    )
    topology.update_node(
        "node3",
        DeviceCapabilities(model="test3", chip="chip3", memory=7000, flops=DeviceFlops(fp32=7, fp16=12, int8=20)),
    )
    topology.add_edge("node1", "node2")
    topology.add_edge("node2", "node3")
    topology.add_edge("node3", "node1")
    topology.add_edge("node1", "node3")

    strategy = HybridFLOPSMemoryPartitioningStrategy()
    partitions = strategy.partition(topology)

    self.assertEqual(len(partitions), 3)
    self.assertEqual(
      partitions = [
        Partition("node3", 0.0, 0.6),  # node3 gets 60% of the layers
        Partition("node1", 0.6, 0.8),  # node1 gets 20% of the layers
        Partition("node2", 0.8, 1.0),  # node2 gets 20% of the layers
      ],
    )

  def test_partition_rounding(self):
    # triangle
    # node1 -> node2 -> node3 -> node1
    topology = Topology()
    topology.update_node(
      "node1",
      DeviceCapabilities(
        model="MacBook Pro",
        chip="test1",
        memory=128*1024*1024*1024,
        flops=DeviceFlops(fp32=4, fp16=6, int8=8),
      ),
    )
    topology.update_node(
      "node2",
      DeviceCapabilities(
        model="Mac Studio",
        chip="test2",
        memory=192*1024*1024*1024,
        fflops=DeviceFlops(fp32=5, fp16=7, int8=10),
      ),
    )
    topology.update_node(
      "node3",
      DeviceCapabilities(
        model="MacBook Pro",
        chip="test3",
        memory=128*1024*1024*1024,
        flops=DeviceFlops(fp32=3, fp16=5, int8=7),
      ),
    )

    strategy = HybridFLOPSMemoryPartitioningStrategy()
    partitions = strategy.partition(topology)

    self.assertEqual(len(partitions), 3)
    self.assertEqual(
      partitions,
      [
        Partition("node2", 0.0, 0.40873),
        Partition("node1", 0.40873, 0.71825),
        Partition("node3", 0.71825, 1.0),
      ],
    )


if __name__ == "__main__":
  unittest.main()