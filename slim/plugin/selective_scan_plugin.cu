// TensorRT plugin "SelectiveScan" (version 1): the forward selective scan of VMamba / SLiM
// (selective_scan_cuda_oflex.fwd with nrows=1 and a float output), running the unmodified upstream kernel from
// third_party/SLiM/src/backbone/vssm/kernels/selective_scan/csrc/selective_scan/cusoflex.
//
// Inputs (linear, contiguous):
//   0 u           [B, D, L]     float or half
//   1 delta       [B, D, L]     same type as u
//   2 A           [D, N]        float
//   3 B           [B, G, N, L]  same type as u
//   4 C           [B, G, N, L]  same type as u
//   5 D_skip      [D]           float
//   6 delta_bias  [D]           float
// Output:
//   0 out         [B, D, L]     float
// Field: delta_softplus (int32, default 1).
#include <cstring>
#include <string>
#include <vector>

#include <NvInferRuntime.h>

#include "selective_scan_fwd_kernel_oflex.cuh"

using namespace nvinfer1;

namespace
{
constexpr char const* kPluginName{"SelectiveScan"};
constexpr char const* kPluginVersion{"1"};
constexpr int kChunkSize = 2048;  // kNThreads * kNItems of the largest forward launch, as in selective_scan_oflex.cpp

class SelectiveScanPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime
{
public:
    explicit SelectiveScanPlugin(int32_t deltaSoftplus)
        : mDeltaSoftplus(deltaSoftplus)
    {
        mFields.emplace_back("delta_softplus", &mDeltaSoftplus, PluginFieldType::kINT32, 1);
        mFieldCollection.nbFields = static_cast<int32_t>(mFields.size());
        mFieldCollection.fields = mFields.data();
    }

    // IPluginV3
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        switch (type)
        {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    IPluginV3* clone() noexcept override
    {
        return new (std::nothrow) SelectiveScanPlugin(mDeltaSoftplus);
    }

    // IPluginV3OneCore
    AsciiChar const* getPluginName() const noexcept override { return kPluginName; }
    AsciiChar const* getPluginVersion() const noexcept override { return kPluginVersion; }
    AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

    // IPluginV3OneBuild
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        return nbInputs == 7 && nbOutputs == 1 ? 0 : -1;
    }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        PluginTensorDesc const& desc = inOut[pos].desc;
        if (desc.format != TensorFormat::kLINEAR)
        {
            return false;
        }
        switch (pos)
        {
        case 0: return desc.type == DataType::kFLOAT || desc.type == DataType::kHALF;
        case 1:
        case 3:
        case 4: return desc.type == inOut[0].desc.type;
        default: return desc.type == DataType::kFLOAT;  // A, D, delta_bias and the output
        }
    }

    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes,
        int32_t nbInputs) const noexcept override
    {
        outputTypes[0] = DataType::kFLOAT;
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override
    {
        outputs[0] = inputs[0];
        return 0;
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override
    {
        Dims const& u = inputs[0].max;
        int64_t const nChunks = (u.d[2] + kChunkSize - 1) / kChunkSize;
        // chunk states x: [B, D, n_chunks, 2 * N] floats
        return static_cast<size_t>(u.d[0] * u.d[1] * nChunks * 2 * inputs[2].max.d[1]) * sizeof(float);
    }

    // IPluginV3OneRuntime
    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        // set_ssm_params_fwd of selective_scan_oflex.cpp for contiguous tensors (strides in elements)
        Dims const& u = inputDesc[0].dims;
        Dims const& b = inputDesc[3].dims;
        int const batch = u.d[0], dim = u.d[1], seqlen = u.d[2];
        int const groups = b.d[1], dstate = b.d[2];

        SSMParamsBase params;
        std::memset(&params, 0, sizeof(params));
        params.batch = batch;
        params.dim = dim;
        params.seqlen = seqlen;
        params.dstate = dstate;
        params.n_groups = groups;
        params.n_chunks = (seqlen + kChunkSize - 1) / kChunkSize;
        params.dim_ngroups_ratio = dim / groups;
        params.dim_deltagroups_ratio = dim / inputDesc[1].dims.d[1];
        params.delta_softplus = mDeltaSoftplus != 0;

        params.u_ptr = const_cast<void*>(inputs[0]);
        params.delta_ptr = const_cast<void*>(inputs[1]);
        params.A_ptr = const_cast<void*>(inputs[2]);
        params.B_ptr = const_cast<void*>(inputs[3]);
        params.C_ptr = const_cast<void*>(inputs[4]);
        params.D_ptr = const_cast<void*>(inputs[5]);
        params.delta_bias_ptr = const_cast<void*>(inputs[6]);
        params.out_ptr = outputs[0];
        params.x_ptr = workspace;

        params.A_d_stride = dstate;
        params.A_dstate_stride = 1;
        params.B_batch_stride = groups * dstate * seqlen;
        params.B_group_stride = dstate * seqlen;
        params.B_dstate_stride = seqlen;
        params.C_batch_stride = groups * dstate * seqlen;
        params.C_group_stride = dstate * seqlen;
        params.C_dstate_stride = seqlen;
        params.u_batch_stride = dim * seqlen;
        params.u_d_stride = seqlen;
        params.delta_batch_stride = inputDesc[1].dims.d[1] * seqlen;
        params.delta_d_stride = seqlen;
        params.out_batch_stride = dim * seqlen;
        params.out_d_stride = seqlen;

        if (inputDesc[0].type == DataType::kHALF)
        {
            selective_scan_fwd_cuda<1, at::Half, float, float>(params, stream);
        }
        else
        {
            selective_scan_fwd_cuda<1, float, float, float>(params, stream);
        }
        return cudaGetLastError() == cudaSuccess ? 0 : -1;
    }

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override { return clone(); }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mFieldCollection; }

private:
    int32_t mDeltaSoftplus;
    std::vector<PluginField> mFields;
    PluginFieldCollection mFieldCollection{};
};

class SelectiveScanCreator : public IPluginCreatorV3One
{
public:
    SelectiveScanCreator()
    {
        mFields.emplace_back("delta_softplus", nullptr, PluginFieldType::kINT32, 1);
        mFieldCollection.nbFields = static_cast<int32_t>(mFields.size());
        mFieldCollection.fields = mFields.data();
    }

    PluginFieldCollection const* getFieldNames() noexcept override { return &mFieldCollection; }

    IPluginV3* createPlugin(AsciiChar const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override
    {
        int32_t deltaSoftplus = 1;
        for (int32_t i = 0; fc != nullptr && i < fc->nbFields; ++i)
        {
            PluginField const& field = fc->fields[i];
            if (std::strcmp(field.name, "delta_softplus") == 0 && field.data != nullptr)
            {
                deltaSoftplus = field.type == PluginFieldType::kINT64 ? static_cast<int32_t>(*static_cast<int64_t const*>(field.data))
                                                                      : *static_cast<int32_t const*>(field.data);
            }
        }
        return new (std::nothrow) SelectiveScanPlugin(deltaSoftplus);
    }

    AsciiChar const* getPluginName() const noexcept override { return kPluginName; }
    AsciiChar const* getPluginVersion() const noexcept override { return kPluginVersion; }
    AsciiChar const* getPluginNamespace() const noexcept override { return ""; }

private:
    std::vector<PluginField> mFields;
    PluginFieldCollection mFieldCollection{};
};
} // namespace

// Entry points of a plugin library for IPluginRegistry::loadLibrary (tensorrt.get_plugin_registry().load_library).
extern "C" void setLoggerFinder(ILoggerFinder* finder) {}

extern "C" IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
{
    static SelectiveScanCreator creator;
    static IPluginCreatorInterface* const creators[] = {&creator};
    nbCreators = 1;
    return creators;
}
