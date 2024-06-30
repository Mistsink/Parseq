# Scene Text Recognition with Permuted Autoregressive Sequence Models

thanks to [Parseq](https://github.com/baudm/parseq)

must refine imgaug.py to work with numpy 2.0

```python
NP_FLOAT_TYPES = {np.float16, np.float32, np.float64, np.float128}
# 检查 float128 的存在性，因为某些平台可能不支持 float128
NP_FLOAT_TYPES = {dtype for dtype in NP_FLOAT_TYPES if hasattr(np, dtype.__name__)}

# 定义整数类型集合
NP_INT_TYPES = {np.int8, np.int16, np.int32, np.int64}
# 检查 int128 的存在性，因为某些平台可能不支持 int128
NP_INT_TYPES = {dtype for dtype in NP_INT_TYPES if hasattr(np, dtype.__name__)}

# 定义无符号整数类型集合
NP_UINT_TYPES = {np.uint8, np.uint16, np.uint32, np.uint64}
# 检查 uint128 的存在性，因为某些平台可能不支持 uint128
NP_UINT_TYPES = {dtype for dtype in NP_UINT_TYPES if hasattr(np, dtype.__name__)}
```
