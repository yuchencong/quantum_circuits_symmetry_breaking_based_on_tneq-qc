"""
Test the new features of TNGraph class.
"""

from tneq_qc.core.tn_graph import TNGraph


print("=" * 70)
print("Test 1: Parse string with no connection (A----B)")
print("=" * 70)

graph1 = TNGraph()
graph1.from_string("-2-A-----B-2-")

print("Input: -2-A-----B-2-")
print(f"Parsed: {graph1.graph[0]}")
print(f"Expected: [('A', 2, 0), ('B', 0, 2)] (0 means no connection)")
print()


print("=" * 70)
print("Test 2: Export (A, 2, 0) (B, 0, 2) to string")
print("=" * 70)

graph2 = TNGraph(n_qubits=1)
graph2.graph[0] = [('A', 2, 0), ('B', 0, 2)]
graph2.tensor_names = ['A', 'B']
graph2.n_tensors = 2

output = graph2.to_string()
print(f"Internal: {graph2.graph[0]}")
print(f"Output: {output}")
print(f"Expected: -2-A-----B-2- (5 dashes between A and B)")
print()


print("=" * 70)
print("Test 3: Export with connection (A, 2, 3) (B, 3, 2)")
print("=" * 70)

graph3 = TNGraph(n_qubits=1)
graph3.graph[0] = [('A', 2, 3), ('B', 3, 2)]
graph3.tensor_names = ['A', 'B']
graph3.n_tensors = 2

output = graph3.to_string()
print(f"Internal: {graph3.graph[0]}")
print(f"Output: {output}")
print(f"Expected: -2-A--3--B-2- (bond 3 centered)")
print()


print("=" * 70)
print("Test 4: Export with 2-digit bond")
print("=" * 70)

graph4 = TNGraph(n_qubits=1)
graph4.graph[0] = [('A', 2, 15), ('B', 15, 2)]
graph4.tensor_names = ['A', 'B']
graph4.n_tensors = 2

output = graph4.to_string()
print(f"Internal: {graph4.graph[0]}")
print(f"Output: {output}")
print(f"Expected: -2-A-15--B-2- (2-digit bond)")
print()


print("=" * 70)
print("Test 5: modify_bond - Change A--3--C to A--4--C")
print("=" * 70)

graph5 = TNGraph(n_qubits=1)
graph5.graph[0] = [('A', 2, 3), ('C', 3, 2)]
graph5.tensor_names = ['A', 'C']
graph5.n_tensors = 2

print(f"Before: {graph5.graph[0]}")
print(f"String: {graph5.to_string()}")

graph5.modify_bond(0, 'A', 4)

print(f"After: {graph5.graph[0]}")
print(f"String: {graph5.to_string()}")
print()


print("=" * 70)
print("Test 6: modify_bond - Change A--3--C to A--0--C (no connection)")
print("=" * 70)

graph6 = TNGraph(n_qubits=1)
graph6.graph[0] = [('A', 2, 3), ('C', 3, 2)]
graph6.tensor_names = ['A', 'C']
graph6.n_tensors = 2

print(f"Before: {graph6.graph[0]}")
print(f"String: {graph6.to_string()}")

graph6.modify_bond(0, 'A', 0)

print(f"After: {graph6.graph[0]}")
print(f"String: {graph6.to_string()}")
print()


print("=" * 70)
print("Test 7: remove_tensor_from_qubit - A--3--B--4--C → A--3--C")
print("=" * 70)

graph7 = TNGraph(n_qubits=1)
graph7.graph[0] = [('A', 2, 3), ('B', 3, 4), ('C', 4, 2)]
graph7.tensor_names = ['A', 'B', 'C']
graph7.n_tensors = 3

print(f"Before: {graph7.graph[0]}")
print(f"String: {graph7.to_string()}")

graph7.remove_tensor_from_qubit(0, 'B')

print(f"After: {graph7.graph[0]}")
print(f"String: {graph7.to_string()}")
print(f"Expected: [('A', 2, 3), ('C', 3, 2)] - uses min(3,4)=3")
print()


print("=" * 70)
print("Test 8: remove_tensor_from_qubit - -2-A--3--B-2- → -2-A-2-")
print("=" * 70)

graph8 = TNGraph(n_qubits=1)
graph8.graph[0] = [('A', 2, 3), ('B', 3, 2)]
graph8.tensor_names = ['A', 'B']
graph8.n_tensors = 2

print(f"Before: {graph8.graph[0]}")
print(f"String: {graph8.to_string()}")

graph8.remove_tensor_from_qubit(0, 'B')

print(f"After: {graph8.graph[0]}")
print(f"String: {graph8.to_string()}")
print(f"Expected: [('A', 2, 2)] - uses edge bond 2")
print()


print("=" * 70)
print("Test 9: insert_tensor_after - A--3--C → A--3--B--3--C")
print("=" * 70)

graph9 = TNGraph(n_qubits=1)
graph9.graph[0] = [('A', 2, 3), ('C', 3, 2)]
graph9.tensor_names = ['A', 'C']
graph9.n_tensors = 2

print(f"Before: {graph9.graph[0]}")
print(f"String: {graph9.to_string()}")

graph9.insert_tensor_after(0, 'A')

print(f"After: {graph9.graph[0]}")
print(f"String: {graph9.to_string()}")
print(f"Expected: [('A', 2, 3), ('B', 3, 3), ('C', 3, 2)]")
print()


print("=" * 70)
print("Test 10: Complex example with all features")
print("=" * 70)

graph10 = TNGraph()
graph10.from_string("""-2-A--3--B--4--C-2-
-2-A--5--B-----C-2-""")

print("Initial graph:")
print(graph10.to_string())
print()

print("Qubit 0:", graph10.graph[0])
print("Qubit 1:", graph10.graph[1])
print()

# Modify bond on qubit 0
print("Modifying B-C connection on qubit 0 to 5...")
graph10.modify_bond(0, 'B', 5)
print(graph10.to_string())
print()

# Remove tensor from qubit 1
print("Removing B from qubit 1...")
graph10.remove_tensor_from_qubit(1, 'B')
print(graph10.to_string())
print()

# Insert tensor
print("Inserting B back between A and C on qubit 1...")
graph10.insert_tensor_after(1, 'A')
print(graph10.to_string())
print()

print("=" * 70)
print("Test 11: Remove only tensor - -2-A-2- → empty line")
print("=" * 70)

graph11 = TNGraph(n_qubits=1)
graph11.graph[0] = [('A', 2, 2)]
graph11.tensor_names = ['A']
graph11.n_tensors = 1

print(f"Before: {graph11.graph[0]}")
print(f"String: {graph11.to_string()}")

graph11.remove_tensor_from_qubit(0, 'A')

print(f"After: {graph11.graph[0]}")
print(f"String: {graph11.to_string()}")
print(f"Expected: [] - empty qubit line")
print()


print("=" * 70)
print("Test 12: Remove left tensor (2 tensors, no connection)")
print("=" * 70)

graph12 = TNGraph(n_qubits=1)
graph12.graph[0] = [('A', 2, 0), ('B', 0, 2)]
graph12.tensor_names = ['A', 'B']
graph12.n_tensors = 2

print(f"Before: {graph12.graph[0]}")
print(f"String: {graph12.to_string()}")

graph12.remove_tensor_from_qubit(0, 'A')

print(f"After: {graph12.graph[0]}")
print(f"String: {graph12.to_string()}")
print(f"Expected: [('B', 2, 2)] - B connects to left edge with bond 2")
print()


print("=" * 70)
print("Test 13: Remove left tensor (2 tensors, with connection)")
print("=" * 70)

graph13 = TNGraph(n_qubits=1)
graph13.graph[0] = [('A', 2, 3), ('B', 3, 2)]
graph13.tensor_names = ['A', 'B']
graph13.n_tensors = 2

print(f"Before: {graph13.graph[0]}")
print(f"String: {graph13.to_string()}")

graph13.remove_tensor_from_qubit(0, 'A')

print(f"After: {graph13.graph[0]}")
print(f"String: {graph13.to_string()}")
print(f"Expected: [('B', 2, 2)] - B connects to left edge with bond 2")
print()


print("=" * 70)
print("Test 14: Change bond from 0 to number - A-----B → A--3--B")
print("=" * 70)

graph14 = TNGraph(n_qubits=1)
graph14.graph[0] = [('A', 2, 0), ('B', 0, 2)]
graph14.tensor_names = ['A', 'B']
graph14.n_tensors = 2

print(f"Before: {graph14.graph[0]}")
print(f"String: {graph14.to_string()}")

graph14.modify_bond(0, 'A', 3)

print(f"After: {graph14.graph[0]}")
print(f"String: {graph14.to_string()}")
print(f"Expected: [('A', 2, 3), ('B', 3, 2)] - connection established with bond 3")
print()


print("=" * 70)
print("Test 15: Change bond from 0 to 2-digit number")
print("=" * 70)

graph15 = TNGraph(n_qubits=1)
graph15.graph[0] = [('A', 2, 0), ('B', 0, 2)]
graph15.tensor_names = ['A', 'B']
graph15.n_tensors = 2

print(f"Before: {graph15.graph[0]}")
print(f"String: {graph15.to_string()}")

graph15.modify_bond(0, 'A', 12)

print(f"After: {graph15.graph[0]}")
print(f"String: {graph15.to_string()}")
print(f"Expected: -2-A-12--B-2- (2-digit bond)")
print()


print("=" * 70)
print("Test 16: Multi-qubit - Remove single tensor from one line")
print("=" * 70)

graph16 = TNGraph()
graph16.from_string("""-2-A--3--B-2-
-2-C-2-
-2-D--4--E-2-""")

print("Before:")
print(graph16.to_string())
print()
print("Qubit 0:", graph16.graph[0])
print("Qubit 1:", graph16.graph[1])
print("Qubit 2:", graph16.graph[2])
print()

print("Removing C from qubit 1...")
graph16.remove_tensor_from_qubit(1, 'C')

print("\nAfter:")
print(graph16.to_string())
print("Qubit 0:", graph16.graph[0])
print("Qubit 1:", graph16.graph[1])
print("Qubit 2:", graph16.graph[2])
print(f"Expected: qubit 1 is now empty []")
print()


print("=" * 70)
print("All tests completed!")
print("=" * 70)
