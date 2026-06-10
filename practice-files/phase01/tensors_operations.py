'''
This files contains the tensor class from scratch, here i learned about how tensors work in actually and how the illusion of
dimentions/axis was implemented in python from a flat list using strides. 
I also practised various Tensor methods like reshpape, squeeze, unsqueeze, permute and transpose.
Here I also implemented einsum() function from scratch and validated it against the np.einsum. 

'''

import numpy as np
from functools import reduce
from itertools import product as iterproduct
import os
import sys

path = "phases/01-math-foundations/12-tensor-operations/code"
sys.path.append(os.path.abspath(path))

from tensors import demo_attention_einsum  # to work with exercise 4


class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)

        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)
            if total != len(self._data):
                raise ValueError(
                    f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )
            self._shape = tuple(shape)
        self._strides = self._compute_strides(self._shape)
    
    def _flatten_nested(self, data):
        if not isinstance(data, (list, tuple)):
            return [data], ()
        if len(data) == 0:
            return [], (0,)
        
        sub_results = [self._flatten_nested(item) for item in data]
        sub_shape = sub_results[0][1]
        for i, (_, s) in enumerate(sub_results):
            if s != sub_shape:
                raise ValueError(
                    f"Inconsistent shapes at index {i}: {s} vs {sub_shape}"
                )

        flat = []
        for sub_data, _ in sub_results:
            flat.extend(sub_data)

        return flat, (len(data),) + sub_shape

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) -2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)
    
    @property
    def shape(self):
        return self._shape

    @property
    def rank(self):
        return len(self._shape)

    @property
    def size(self):
        return len(self._data)

    @property
    def strides(self):
        return self._strides
    
    def _flat_index(self, indices):
        if len(indices) != len(self._shape):
            raise IndexError(
                f"Expected {len(self._shape)} indices, got {len(indices)}"
            )
        idx = 0
        for i, (ind, stride) in enumerate(zip(indices, self._strides)):
            if ind < 0 or ind >= self._shape[i]:
                raise IndexError(
                    f"Index {ind} out of range for axis {i} with size {self._shape[i]}"
                )
            idx += ind * stride
        return idx
    
    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) == len(self._shape):
            return self._data[self._flat_index(indices)]
        raise IndexError("Partial indexing not supported in this basic implementation")

    def __setitem__(self, indices, value):
        if not isinstance(indices, tuple):
            indices = (indices,)
        self._data[self._flat_index(indices)] = value

    def reshape(self, new_shape):
        new_shape = list(new_shape)
        neg_idx = -1
        known_product = 1
        for i, s in enumerate(new_shape):
            if s == -1:
                if neg_idx != -1:
                    raise ValueError("Only one dimension can be -1")
                neg_idx = i
            else:
                known_product *= s

        if neg_idx != -1:
            new_shape[neg_idx] = self.size // known_product

        total = reduce(lambda a, b: a * b, new_shape, 1)
        if total != self.size:
            raise ValueError(
                f"Cannot reshape {self.size} elements into shape {tuple(new_shape)}"
            )

        result = Tensor.__new__(Tensor)
        result._data = self._data[:]
        result._shape = tuple(new_shape)
        result._strides = self._compute_strides(result._shape)
        return result
    
    def squeeze(self, dim=None):
        if dim is not None:
            if self._shape[dim] != 1:
                return self.reshape(self._shape)
            new_shape = list(self._shape)
            new_shape.pop(dim)
            return self.reshape(tuple(new_shape) if new_shape else ())
        new_shape = tuple(s for s in self._shape if s != 1)
        if not new_shape:
            new_shape = ()
        return self.reshape(new_shape)

    def unsqueeze(self, dim):
        if dim < 0:
            dim = len(self._shape) + 1 + dim
        new_shape = list(self._shape)
        new_shape.insert(dim, 1)
        return self.reshape(tuple(new_shape))

    def transpose(self, dim0, dim1):
        perm = list(range(self.rank))
        perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return self.permute(perm)

    def permute(self, dims):
        if sorted(dims) != list(range(self.rank)):
            raise ValueError(f"Invalid permutation {dims} for rank {self.rank}")

        new_shape = tuple(self._shape[d] for d in dims)
        result = Tensor.__new__(Tensor)
        result._shape = new_shape
        result._strides = self._compute_strides(new_shape)
        result._data = [0] * self.size

        old_strides = self._strides
        for old_indices in iterproduct(*(range(s) for s in self._shape)):
            new_indices = tuple(old_indices[d] for d in dims)
            old_flat = sum(i * s for i, s in zip(old_indices, old_strides))
            new_flat = sum(
                i * s for i, s in zip(new_indices, result._strides)
            )
            result._data[new_flat] = self._data[old_flat]

        return result

    def flatten(self, start_dim=0, end_dim=-1):
        if end_dim < 0:
            end_dim = self.rank + end_dim
        new_shape = (
            list(self._shape[:start_dim])
            + [reduce(lambda a, b: a * b, self._shape[start_dim:end_dim + 1], 1)]
            + list(self._shape[end_dim + 1:])
        )
        return self.reshape(tuple(new_shape))

    def _elementwise_op(self, other, op):
        if isinstance(other, (int, float)):
            result_data = [op(x, other) for x in self._data]
            return Tensor(result_data, shape=self._shape)
        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type {type(other)}")
        
        ## New auto-broadcasting logic
        tensor_a = self
        tensor_b = other

        if tensor_a._shape != tensor_b.shape:
            # determine the maximum target shape for the each axis
            target_shape = tuple(
                max(s_a, s_b) for s_a, s_b in zip(tensor_a._shape, tensor_b._shape)
            )
            tensor_a = tensor_a.broadcast_to(target_shape)
            tensor_b = tensor_b.broadcast_to(target_shape)

        # if self._shape != other._shape:
        #     raise ValueError(
        #         f"Shape mismatch: {self._shape} vs {other._shape}. "
        #         "Use broadcast() first."
        #     )
        result_data = [op(a, b) for a, b in zip(tensor_a._data, tensor_b._data)]
        return Tensor(result_data, shape=tensor_a._shape)
    
    ## Exercise 3 implemented here:
    def broadcast_to(self, target_shape):
        target_shape = tuple(target_shape)

        if len(self._shape) != len(target_shape):
            raise ValueError("Rank mismatch! Broadcasting requires equal ranks in this implementation.")
        
        for s, t in zip(self._shape, target_shape):
            if s != t and s != 1:
                raise ValueError(f"Cannot broadcast axis of size {s} to size {t}")
        
        # Build a brand new tensor object from scratch
        result = Tensor.__new__(Tensor)
        result._shape = target_shape
        result._strides = self._compute_strides(target_shape)
        result._data = []  # we'll fill this flat list up element-by-element.

        # Core logic: loop through every target coordinate
        for target_indices in iterproduct(*(range(s) for s in target_shape)):

            orig_indices = tuple(
                0 if self._shape[i] == 1 else target_indices[i]
                for i in range(len(target_shape))
            )
            # using our existing _flat_index map to find where that element is in our flast list
            old_flat_idx = self._flat_index(orig_indices)
            result._data.append(self._data[old_flat_idx])
        return result

    def __add__(self, other):
        return self._elementwise_op(other, lambda a, b: a + b)

    def __mul__(self, other):
        return self._elementwise_op(other, lambda a, b: a * b)

    def __sub__(self, other):
        return self._elementwise_op(other, lambda a, b: a - b)

    def sum(self, axis=None):
        if axis is None:
            return sum(self._data)
        if axis < 0:
            axis = self.rank + axis
        new_shape = list(self._shape)
        axis_size = new_shape.pop(axis)

        result_size = reduce(lambda a, b: a * b, new_shape, 1)
        result_data = [0.0] * result_size
        result_strides = self._compute_strides(tuple(new_shape))

        for indices in iterproduct(*(range(s) for s in self._shape)):
            old_flat = sum(i * s for i, s in zip(indices, self._strides))
            new_indices = indices[:axis] + indices[axis + 1:]
            if new_indices:
                new_flat = sum(
                    i * s for i, s in zip(new_indices, result_strides)
                )
            else:
                new_flat = 0
            result_data[new_flat] += self._data[old_flat]

        if not new_shape:
            return result_data[0]
        return Tensor(result_data, shape=tuple(new_shape))


    def to_list(self):
        if self.rank == 0:
            return self._data[0]
        return self._build_nested(self._data, self._shape, 0)

    def _build_nested(self, data, shape, offset):
        if len(shape) == 1:
            return data[offset:offset + shape[0]]
        result = []
        stride = reduce(lambda a, b: a * b, shape[1:], 1)
        for i in range(shape[0]):
            result.append(self._build_nested(data, shape[1:], offset + i * stride))
        return result

    def __repr__(self):
        return f"Tensor(shape={self._shape}, data={self.to_list()})"

    def to_numpy(self):
        return np.array(self._data).reshape(self._shape)

## Exercise 3: Hard -- Build einsum from scratch. Implement a basic einsum(subscripts, *tensors) function that handles at least: 
# dot product (i,i->), matrix multiply (ij,jk->ik), outer product (i,j->ij), and transpose (ij->ji). Parse the subscript string, 
# identify contracted indices, and loop over all index combinations. Compare your results against np.einsum.


def einsum(subscripts, *tensors):
    # parsing the subscripts string into input labels and output labels
    input_str, output_str = subscripts.replace(" ", "").split("->")
    input_labels = input_str.split(",")

    # map every unique label to its actual geometric size
    label_to_size = {}
    for labels, tensor in zip(input_labels, tensors):
        for char, size in zip(labels, tensor.shape):
            if char in label_to_size and label_to_size[char] != size:
                raise ValueError(f"Size mismatch for axis '{char}': {label_to_size[char]} vs {size}")
            label_to_size[char] = size

    # figure out all unique labels across the entire operation in a fixed order
    all_unique_labels = sorted(list(label_to_size.keys()))

    # determine the shape of our final output tensor
    output_shape = tuple(label_to_size[char] for char in output_str)

    # create our blank target output Tensor instance
    result = Tensor.__new__(Tensor)
    result._shape = output_shape
    result._strides = Tensor._compute_strides(output_shape)
    result._data = [0.0] * (reduce(lambda a, b: a * b, output_shape, 1) if output_shape else 1)
    # 5. THE MEGA LOOP: Generate every single coordinate combination across all axes
    # For matrix multiply with shapes (2,3) and (3,4), this loops over every (i, j, k) combo!
    for global_indices in iterproduct(*(range(label_to_size[char]) for char in all_unique_labels)):
        
        # Make a fast lookup map for the current loop iteration: e.g., {'i': 0, 'j': 1, 'k': 2}
        current_coords = dict(zip(all_unique_labels, global_indices))
        
        # Pull out the exact local element coordinates for each input tensor
        val = 1.0
        for labels, tensor in zip(input_labels, tensors):
            tensor_coords = tuple(current_coords[char] for char in labels)
            # Use our existing flat index project logic to sample the number!
            val *= tensor._data[tensor._flat_index(tensor_coords)]
            
        # Determine exactly where this product needs to be accumulated in the output tensor
        out_coords = tuple(current_coords[char] for char in output_str)
        if out_coords:
            out_flat_idx = result._flat_index(out_coords)
            result._data[out_flat_idx] += val
        else:
            # Handling scalar dot product outputs (i, i -> )
            result._data[0] += val

    # If the output shape is completely empty, just return the raw scalar number
    if not output_shape:
        return result._data[0]
        
    return result


# Setup test variables
a = Tensor([1, 2, 3], shape=(3,))
b = Tensor([4, 5, 6], shape=(3,))
x = Tensor([[1, 2], [3, 4]], shape=(2, 2))
y = Tensor([[5, 6], [7, 8]], shape=(2, 2))

print("--- RUNNING EINSUM TESTING SUITE ---")

# 1. Dot Product (i, i -> )
# Expected: (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
custom_dot = einsum("i,i->", a, b)
np_dot = np.einsum("i,i->", a.to_numpy(), b.to_numpy())
print(f"Dot Product Match: Custom={custom_dot} | NumPy={np_dot}")

# 2. Transpose (ij -> ji)
custom_trans = einsum("ij->ji", x)
print(f"Transpose Shape: {custom_trans.shape} | Matrix:\n{custom_trans.to_numpy()}")

# 3. Matrix Multiplication (ij, jk -> ik)
custom_matmul = einsum("ij,jk->ik", x, y)
np_matmul = np.einsum("ij,jk->ik", x.to_numpy(), y.to_numpy())
print(f"\nMatMul Match:\nCustom:\n{custom_matmul.to_numpy()}\nNumPy:\n{np_matmul}")

# 4. Outer Product (i, j -> ij)
custom_outer = einsum("i,j->ij", a, b)
print(f"\nOuter Product Shape: {custom_outer.shape}\nMatrix:\n{custom_outer.to_numpy()}")

## Exercise 4: Hard -- Attention shape tracker. Write a function that takes batch_size, seq_len, embed_dim, and num_heads 
# as inputs and prints the exact shape at every step of multi-head attention: input, Q/K/V projection, head split, attention scores, 
# softmax weights, weighted sum, head merge, output projection. Verify against the demo_attention_einsum() output.

def ast(batch_size, seq_len, embed_dim, num_heads):

    head_dim = embed_dim // num_heads
    print(f"config: batch={batch_size}, seqlen={seq_len}, embed demention={embed_dim}, num heads={num_heads}, head dim={head_dim}")

    # step1: input: shape(B, T, E)
    print(f"input X : ({batch_size}, {seq_len}, {embed_dim})")

    # step 2: Q/K/V projections
    # projection weights are (E, E). Multiplied by X (B, T, E), they output (B, T, E)
    print(f"2. Q/K/V projections:  ({batch_size}, {seq_len}, {embed_dim})")

    # step 3: Head split and transpose --
    # First we reshape, (B, T, E) -> (B, T, H, D)
    print(f". Head Split (Q, K, V):  ({batch_size}, {num_heads}, {seq_len}, {head_dim})")

    # step 4: attention scores
    # here we multiply Q (B, H, T, D) by K (B, H, S, D)
    print(f"attention scores:    ({batch_size}, {num_heads}, {seq_len}, {seq_len})")

    # step 5: Softmax weights
    print(f"Softmax weights:    ({batch_size}, {num_heads}, {seq_len}, {seq_len})")

    # step 6: Weighted sum(Context Vector)
    # Here we multiply weights (B, H, T, S) by V (B, H, S, D).. 'S' is contracted.
    print(f"Attention value output:  ({batch_size}, {num_heads}, {seq_len}, {head_dim})")

    # step 7: Head Merge(Concatenation):
    # Transpose back: (B, H, T, D) -> (B, T, H, D)
    print(f"Concatenated heads:     ({batch_size}, {seq_len}, {embed_dim})")

    # step 8: output Projection ---
    # Final layer weights W_o (E, E) multiply by our merged heads (B, T, E) -> (B, T, E)
    print(f"8. Final Output:     {batch_size}, {seq_len}, {embed_dim}")


ast(batch_size=2, seq_len=8, embed_dim=64, num_heads=4)
demo_attention_einsum()  # validating it against the demo_attention_einsum() from tutorial.