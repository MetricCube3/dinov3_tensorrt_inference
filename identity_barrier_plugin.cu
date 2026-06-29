// IdentityBarrier: a no-op passthrough TensorRT plugin.
//
// Purpose: a plugin node is OPAQUE to TensorRT's Myelin fusion compiler, which
// therefore cannot fuse across it. By inserting these identity barriers on the
// inputs/outputs of the attention matmuls, we force QK^T / softmax / attn@V to
// be built as plain (correct) TensorRT layers instead of Myelin's buggy fused
// attention region.
//
// The kernel is a device-to-device copy: output == input, bit for bit.
//
// Build (on the Jetson):
//   nvcc -shared -std=c++17 -o libidentity_barrier.so identity_barrier_plugin.cu \
//        -Xcompiler -fPIC \
//        -I/usr/include/aarch64-linux-gnu \
//        -L/usr/lib/aarch64-linux-gnu -lnvinfer -lcudart

#include "NvInfer.h"
#include <cuda_runtime.h>
#include <string>

using namespace nvinfer1;

namespace {
constexpr char const* IDB_NAME = "IdentityBarrier";
constexpr char const* IDB_VERSION = "1";
}  // namespace

class IdentityBarrier : public IPluginV2DynamicExt {
public:
    IdentityBarrier() = default;
    IdentityBarrier(void const*, size_t) {}

    // ---- IPluginV2DynamicExt ----
    IPluginV2DynamicExt* clone() const noexcept override {
        auto* p = new IdentityBarrier();
        p->setPluginNamespace(mNs.c_str());
        return p;
    }

    DimsExprs getOutputDimensions(int32_t, DimsExprs const* inputs, int32_t,
                                  IExprBuilder&) noexcept override {
        return inputs[0];
    }

    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* inOut,
                                   int32_t, int32_t) noexcept override {
        PluginTensorDesc const& d = inOut[pos];
        bool const okFmt = d.format == TensorFormat::kLINEAR;
        if (pos == 0) {
            return okFmt && (d.type == DataType::kFLOAT || d.type == DataType::kHALF);
        }
        // output: must match the input type
        return okFmt && (d.type == inOut[0].type);
    }

    void configurePlugin(DynamicPluginTensorDesc const*, int32_t,
                         DynamicPluginTensorDesc const*, int32_t) noexcept override {}

    size_t getWorkspaceSize(PluginTensorDesc const*, int32_t,
                            PluginTensorDesc const*, int32_t) const noexcept override {
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void*,
                    cudaStream_t stream) noexcept override {
        size_t vol = 1;
        for (int32_t i = 0; i < inputDesc[0].dims.nbDims; ++i) {
            vol *= static_cast<size_t>(inputDesc[0].dims.d[i]);
        }
        size_t const elemSize = (inputDesc[0].type == DataType::kHALF) ? 2 : 4;
        cudaMemcpyAsync(outputs[0], inputs[0], vol * elemSize,
                        cudaMemcpyDeviceToDevice, stream);
        return 0;
    }

    DataType getOutputDataType(int32_t, DataType const* inputTypes,
                               int32_t) const noexcept override {
        return inputTypes[0];
    }

    // ---- IPluginV2 ----
    char const* getPluginType() const noexcept override { return IDB_NAME; }
    char const* getPluginVersion() const noexcept override { return IDB_VERSION; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override { return 0; }
    void serialize(void*) const noexcept override {}
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(char const* ns) noexcept override { mNs = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return mNs.c_str(); }

private:
    std::string mNs;
};

class IdentityBarrierCreator : public IPluginCreator {
public:
    IdentityBarrierCreator() {
        mFC.nbFields = 0;
        mFC.fields = nullptr;
    }
    char const* getPluginName() const noexcept override { return IDB_NAME; }
    char const* getPluginVersion() const noexcept override { return IDB_VERSION; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }

    IPluginV2* createPlugin(char const*, PluginFieldCollection const*) noexcept override {
        auto* p = new IdentityBarrier();
        p->setPluginNamespace(mNs.c_str());
        return p;
    }
    IPluginV2* deserializePlugin(char const*, void const* data, size_t length) noexcept override {
        auto* p = new IdentityBarrier(data, length);
        p->setPluginNamespace(mNs.c_str());
        return p;
    }
    void setPluginNamespace(char const* ns) noexcept override { mNs = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return mNs.c_str(); }

private:
    PluginFieldCollection mFC{};
    std::string mNs;
};

REGISTER_TENSORRT_PLUGIN(IdentityBarrierCreator);
