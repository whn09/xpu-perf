## ccl_ops

### [all_reduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)
```python
torch.distributed.all_reduce(tensor, op=<RedOpType.SUM: 0>, group=None, async_op=False)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size, dim_size] |



### [reduce_scatter](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter_tensor)
```python
torch.distributed.reduce_scatter_tensor(output, input, op=<RedOpType.SUM: 0>, group=None, async_op=False)[source]
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size // world_size, dim_size] |



### [all_gather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_into_tensor)
```python
torch.distributed.all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size // world_size, dim_size] |
| output_tensor | [batch_size, dim_size] |


### [all_to_all](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single)
```python
torch.distributed.all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False)[source]
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size, dim_size] |



### [broadcast](https://pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast)
```python
for i in range(world_size):
    torch.distributed.broadcast(src[i], src=i, group=None, async_op=False)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |

One broadcast per rank, so the measured latency covers `world_size` of them;
`summary()` divides latency and both sizes by `world_size` to report a
per-broadcast figure. Unlike the four ops above, `bus_size == algo_size` — there
is no `2(n-1)/n` factor — so this row's GB/s is comparable to their *algo_bw*,
not to their `bus_bw`.


### all_reduce_h2d
```python
torch.distributed.all_reduce(data, async_op=True)   # default group, never waited
dst.copy_(src)                                      # host -> device, pinned src
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| data | [batch_size, dim_size] |
| src (host) | [batch_size, dim_size] |
| dst (device) | [batch_size, dim_size] |

Measures *overlap* of a collective with a host-to-device copy, not a collective.
Two things to know before quoting a number from it: the `all_reduce` goes to the
**default** group rather than `op_group`, so it always spans every rank the launch
started while `world_size` in the workload file only feeds the `bus_size` formula;
and `async_op=True` is never `wait()`ed, so the timing window catches an
unspecified fraction of the collective.


### [p2p](https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend)
```python
torch.distributed.isend(tensor, dst, tag=0, group=None, async_op=False)
torch.distributed.irecv(tensor, src, tag=0, group=None, async_op=False)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size, dim_size] |

Given a set of devices, we perform pairwise bandwidth tests between every two devices to evaluate the link bandwidth performance of the entire system topology. The test configuration is fixed with a tensor shape of [1024, 2097152], a data type of int8, and a total data volume of 2 GiB.






