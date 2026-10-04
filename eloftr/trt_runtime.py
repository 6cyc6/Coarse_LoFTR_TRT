"""TensorRT engine building and execution with torch CUDA tensors as I/O buffers (TensorRT 10.x/11.x API)."""
from pathlib import Path

import tensorrt as trt
import torch

TRT_MAJOR = int(trt.__version__.split('.')[0])

_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.INT8: torch.int8,
    trt.DataType.UINT8: torch.uint8,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.BF16: torch.bfloat16,
}


def _logger(verbose=False):
    return trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)


def build_engine(onnx_path, engine_path, strongly_typed=True, fp16=False, tf32=False, workspace_gib=4.0,
                 opt_level=3, timing_cache=None, verbose=False):
    """Build and save a TensorRT engine from an ONNX file.

    Args:
        strongly_typed: take tensor types from the ONNX graph (how mixed precision is expressed for
            EfficientLoFTR; the only mode in TensorRT 11)
        fp16: weakly-typed FP16 builder flag for untyped legacy ONNX models (TensorRT 10 only)
        tf32: allow TF32 for fp32 math; off by default because it lowers fp32 layers to ~fp16 mantissa
    """
    logger = _logger(verbose)
    builder = trt.Builder(logger)
    flags = 0
    if strongly_typed and TRT_MAJOR < 11:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f'Failed to parse {onnx_path}:\n{errors}')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30)))
    config.builder_optimization_level = opt_level
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    if fp16:
        if strongly_typed:
            raise ValueError('The fp16 builder flag only applies to weakly-typed networks')
        if TRT_MAJOR >= 11:
            raise ValueError('TensorRT 11 removed the fp16 builder flag; export a typed ONNX model instead')
        config.set_flag(trt.BuilderFlag.FP16)

    cache = None
    if timing_cache is not None:
        timing_cache = Path(timing_cache)
        cache = config.create_timing_cache(timing_cache.read_bytes() if timing_cache.exists() else b'')
        config.set_timing_cache(cache, ignore_mismatch=False)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f'TensorRT failed to build an engine from {onnx_path}')
    Path(engine_path).write_bytes(bytes(serialized))
    if cache is not None:
        timing_cache.write_bytes(bytes(cache.serialize()))
    return Path(engine_path)


class TRTEngine:
    """Static-shape TensorRT engine whose inputs and outputs are preallocated torch CUDA tensors.

    The engine runs on its own stream, ordered after and before the caller's current torch stream.
    Outputs returned by __call__ are the engine's buffers and are overwritten by the next call.
    """

    def __init__(self, engine_path, device='cuda'):
        self.device = torch.device(device)
        torch.cuda.init()
        logger = _logger()
        trt.init_libnvinfer_plugins(logger, '')
        self.runtime = trt.Runtime(logger)
        with torch.cuda.device(self.device):
            self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
            if self.engine is None:
                raise RuntimeError(f'Failed to load {engine_path}; engines only load with the TensorRT version '
                                   f'that built them (this is {trt.__version__})')
            self.context = self.engine.create_execution_context()
        self.inputs, self.outputs = {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise ValueError(f'Dynamic shape for {name}: {shape}; only static engines are supported')
            buffer = torch.empty(shape, dtype=_TO_TORCH[self.engine.get_tensor_dtype(name)], device=self.device)
            self.context.set_tensor_address(name, buffer.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs[name] = buffer
            else:
                self.outputs[name] = buffer
        self.stream = torch.cuda.Stream(self.device)  # TensorRT adds synchronizations on the default stream
        self.graph = None

    def _enqueue(self):
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')

    def capture_cuda_graph(self):
        """Record one engine execution in a CUDA graph to cut per-call launch overhead."""
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        self._enqueue()  # TensorRT needs one regular execution before capture
        self.stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=self.stream):
            self._enqueue()
        self.stream.synchronize()
        self.graph = graph

    def __call__(self, *args, **kwargs):
        """Copy inputs (positional in engine order, or by name) into the engine buffers and execute."""
        feeds = dict(zip(self.inputs, args), **kwargs)
        if feeds.keys() != self.inputs.keys():
            raise ValueError(f'Expected inputs {list(self.inputs)}, got {list(feeds)}')
        for name, value in feeds.items():
            buffer = self.inputs[name]
            value = torch.as_tensor(value).to(device=self.device, dtype=buffer.dtype)
            buffer.copy_(value.reshape(buffer.shape))
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._enqueue()
        current.wait_stream(self.stream)
        return self.outputs
