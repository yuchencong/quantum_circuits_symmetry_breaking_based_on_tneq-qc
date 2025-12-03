"""
Test for GreedyStrategy contraction.

This test verifies the greedy contraction strategy works correctly.
"""

import torch
import numpy as np
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN, QCTNHelper
from tneq_qc.contractor import GreedyStrategy, StrategyCompiler


class TestGreedyStrategy:
    """Test class for GreedyStrategy"""
    
    @staticmethod
    def create_simple_qctn():
        """Create a simple QCTN for testing"""
        # Simple chain: -3-A-3-B-3-
        graph = "-3-A-3-B-3-"
        backend = BackendFactory.create_backend('pytorch')
        qctn = QCTN(graph, backend=backend)
        return qctn
    
    @staticmethod
    def create_multi_qubit_qctn():
        """Create a multi-qubit QCTN for testing"""
        # 3 qubits chain structure
        graph = "-3-a------------3-\n" \
                "-3-a--3--b------3-\n" \
                "-3-------b------3-"
        backend = BackendFactory.create_backend('pytorch')
        qctn = QCTN(graph, backend=backend)
        return qctn
    
    @staticmethod
    def create_standard_graph_qctn(n=3):
        """Create standard graph QCTN"""
        graph = QCTNHelper.generate_example_graph()
        backend = BackendFactory.create_backend('pytorch')
        qctn = QCTN(graph, backend=backend)
        return qctn

    @staticmethod
    def test_strategy_check_compatibility():
        """Test that GreedyStrategy always returns True for compatibility"""
        strategy = GreedyStrategy()
        qctn = TestGreedyStrategy.create_simple_qctn()
        
        shapes_info = {
            'circuit_states_shapes': [(3,)],
            'measure_shapes': [(10, 3, 3)],
        }
        
        result = strategy.check_compatibility(qctn, shapes_info)
        assert result == True, "GreedyStrategy should always be compatible"
        print("✓ check_compatibility test passed")
    
    @staticmethod
    def test_strategy_name():
        """Test that strategy name is correct"""
        strategy = GreedyStrategy()
        assert strategy.name == "greedy", f"Expected 'greedy', got '{strategy.name}'"
        print("✓ strategy name test passed")
    
    @staticmethod
    def test_strategy_estimate_cost():
        """Test cost estimation"""
        strategy = GreedyStrategy()
        qctn = TestGreedyStrategy.create_simple_qctn()
        
        shapes_info = {}
        cost = strategy.estimate_cost(qctn, shapes_info)
        
        assert cost == 5e5, f"Expected 5e5, got {cost}"
        print("✓ estimate_cost test passed")
    
    @staticmethod
    def test_adjacency_table_structure():
        """Test that adjacency_table is correctly built"""
        qctn = TestGreedyStrategy.create_simple_qctn()
        
        print("\n--- Adjacency Table Structure ---")
        for core_info in qctn.adjacency_table:
            print(f"Core {core_info['core_name']} (idx {core_info['core_idx']}):")
            print(f"  input_shape: {core_info['input_shape']}")
            print(f"  output_shape: {core_info['output_shape']}")
            print(f"  input_dim: {core_info['input_dim']}")
            print(f"  output_dim: {core_info['output_dim']}")
            print(f"  in_edge_list: {core_info['in_edge_list']}")
            print(f"  out_edge_list: {core_info['out_edge_list']}")
        
        # Verify structure
        assert len(qctn.adjacency_table) == qctn.ncores
        for core_info in qctn.adjacency_table:
            assert 'core_idx' in core_info
            assert 'core_name' in core_info
            assert 'in_edge_list' in core_info
            assert 'out_edge_list' in core_info
            assert 'input_shape' in core_info
            assert 'output_shape' in core_info
            assert 'input_dim' in core_info
            assert 'output_dim' in core_info
        
        print("✓ adjacency_table structure test passed")
    
    @staticmethod
    def test_compute_function_simple():
        """Test compute function with simple graph"""
        print("\n--- Test Compute Function (Simple) ---")
        
        qctn = TestGreedyStrategy.create_simple_qctn()
        strategy = GreedyStrategy()
        backend = qctn.backend
        
        # Prepare inputs
        nqubits = qctn.nqubits
        batch_size = 5
        
        # Circuit states: one vector per qubit
        circuit_states = [torch.randn(3) for _ in range(nqubits)]
        
        # Measurement matrices: (B, d, d) shape
        measure_matrices = [torch.randn(batch_size, 3, 3) for _ in range(nqubits)]
        
        circuit_states = [backend.convert_to_tensor(s) for s in circuit_states]
        measure_matrices = [backend.convert_to_tensor(m) for m in measure_matrices]

        shapes_info = {
            'circuit_states_shapes': tuple(s.shape for s in circuit_states),
            'measure_shapes': tuple(m.shape for m in measure_matrices),
        }

        # Get compute function
        compute_fn = strategy.get_compute_function(qctn, shapes_info, backend)
        
        # Execute
        result = compute_fn(qctn.cores_weights, circuit_states, measure_matrices)
        
        print(f"Result shape: {result.shape}")
        # print(f"Result: {result}")
        
        print("✓ compute_function simple test passed")
        return result
    
    @staticmethod
    def test_compute_function_multi_qubit():
        """Test compute function with multi-qubit graph"""
        print("\n--- Test Compute Function (Multi-Qubit) ---")
        
        qctn = TestGreedyStrategy.create_multi_qubit_qctn()
        strategy = GreedyStrategy()
        backend = qctn.backend
        
        print(f"Graph:\n{qctn.graph}")
        print(f"nqubits: {qctn.nqubits}, ncores: {qctn.ncores}")
        
        # Prepare inputs
        nqubits = qctn.nqubits
        batch_size = 5
        
        # Circuit states: one vector per qubit
        circuit_states = [torch.randn(3) for _ in range(nqubits)]
        
        # Measurement matrices: (B, d, d) shape
        measure_matrices = [torch.randn(batch_size, 3, 3) for _ in range(nqubits)]
        
        circuit_states = [backend.convert_to_tensor(s) for s in circuit_states]
        measure_matrices = [backend.convert_to_tensor(m) for m in measure_matrices]

        shapes_info = {
            'circuit_states_shapes': tuple(s.shape for s in circuit_states),
            'measure_shapes': tuple(m.shape for m in measure_matrices),
        }
        
        # Get compute function
        compute_fn = strategy.get_compute_function(qctn, shapes_info, backend)
        
        # Execute
        result = compute_fn(qctn.cores_weights, circuit_states, measure_matrices)
        
        print(f"Result shape: {result.shape}")
        # print(f"Result: {result}")
        
        print("✓ compute_function multi-qubit test passed")
        return result
    
    @staticmethod
    def test_compute_function_conditional_measure():
        """Test compute function with conditional measurement matrices"""
        print("\n--- Test Compute Function (Conditional Measurement) ---")
        
        qctn = TestGreedyStrategy.create_simple_qctn()
        strategy = GreedyStrategy()
        backend = qctn.backend
        
        # Prepare inputs
        nqubits = qctn.nqubits
        batch_size = 5
        
        # Circuit states
        circuit_states = [torch.randn(3) for _ in range(nqubits)]
        
        # Conditional measurement matrices: (B, 2, d, d) shape
        measure_matrices = [torch.randn(batch_size, 2, 3, 3) for _ in range(nqubits)]
        
        circuit_states = [backend.convert_to_tensor(s) for s in circuit_states]
        measure_matrices = [backend.convert_to_tensor(m) for m in measure_matrices]

        shapes_info = {
            'circuit_states_shapes': tuple(s.shape for s in circuit_states),
            'measure_shapes': tuple(m.shape for m in measure_matrices),
        }
        
        # Get compute function
        compute_fn = strategy.get_compute_function(qctn, shapes_info, backend)
        
        # Execute
        result = compute_fn(qctn.cores_weights, circuit_states, measure_matrices)
        
        print(f"Result shape: {result.shape}")
        # print(f"Result: {result}")
        
        print("✓ compute_function conditional measurement test passed")
        return result
    
    @staticmethod
    def test_strategy_registration():
        """Test that GreedyStrategy is registered in compiler"""
        print("\n--- Test Strategy Registration ---")
        
        strategies = StrategyCompiler.get_registered_strategies()
        
        assert 'greedy' in strategies, "GreedyStrategy should be registered"
        assert isinstance(strategies['greedy'], GreedyStrategy)
        
        # Check it's in balanced and full modes
        assert 'greedy' in StrategyCompiler.MODES['balanced']
        assert 'greedy' in StrategyCompiler.MODES['full']
        
        print(f"Registered strategies: {list(strategies.keys())}")
        print(f"Balanced mode strategies: {StrategyCompiler.MODES['balanced']}")
        print(f"Full mode strategies: {StrategyCompiler.MODES['full']}")
        
        print("✓ strategy registration test passed")
    
    @staticmethod
    def test_compiler_with_greedy():
        """Test StrategyCompiler selects GreedyStrategy correctly"""
        print("\n--- Test Compiler with Greedy ---")
        
        qctn = TestGreedyStrategy.create_simple_qctn()
        backend = qctn.backend
        
        shapes_info = {
            'circuit_states_shapes': ((3,),),
            'measure_shapes': ((5, 3, 3),),
        }
        
        # Use balanced mode which includes greedy
        compiler = StrategyCompiler(mode='balanced')
        
        compute_fn, strategy_name, cost = compiler.compile(qctn, shapes_info, backend)
        
        print(f"Selected strategy: {strategy_name}")
        print(f"Estimated cost: {cost}")
        
        print("✓ compiler with greedy test passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running GreedyStrategy Tests")
    print("=" * 60)
    
    TestGreedyStrategy.test_strategy_name()
    TestGreedyStrategy.test_strategy_check_compatibility()
    TestGreedyStrategy.test_strategy_estimate_cost()
    TestGreedyStrategy.test_adjacency_table_structure()
    TestGreedyStrategy.test_strategy_registration()
    
    print("\n" + "=" * 60)
    print("Running Compute Function Tests")
    print("=" * 60)
    
    # These tests may fail if the implementation has issues
    try:
        TestGreedyStrategy.test_compute_function_simple()
    except Exception as e:
        print(f"✗ compute_function simple test failed: {e}")

    try:
        TestGreedyStrategy.test_compute_function_multi_qubit()
    except Exception as e:
        print(f"✗ compute_function multi-qubit test failed: {e}")

    try:
        TestGreedyStrategy.test_compute_function_conditional_measure()
    except Exception as e:
        print(f"✗ compute_function conditional measurement test failed: {e}")
    
    try:
        TestGreedyStrategy.test_compiler_with_greedy()
    except Exception as e:
        print(f"✗ compiler with greedy test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Tests Complete")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
