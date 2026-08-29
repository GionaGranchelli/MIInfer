#include "miinfer/device_validation.hpp"

#include "miinfer/hip_check.hpp"

#include <hip/hip_runtime_api.h>

#include <ostream>

namespace miinfer {

namespace {

bool inspect_device(int index, DeviceInfo& device, std::string& error) {
    hipDeviceProp_t properties{};
    const hipError_t result = hipGetDeviceProperties(&properties, index);
    if (result != hipSuccess) {
        error = "hipGetDeviceProperties failed for device " + std::to_string(index) + ": "
                + hipGetErrorString(result);
        return false;
    }

    device.index = index;
    device.name = properties.name;
    device.architecture = properties.gcnArchName;
    device.total_vram_bytes = properties.totalGlobalMem;
    return true;
}

bool is_gfx906(const DeviceInfo& device) {
    return device.architecture == "gfx906" || device.architecture.rfind("gfx906:", 0) == 0;
}

}  // namespace

bool validate_gfx906_device(int requested_index, DeviceInfo& device, std::string& error) {
    int device_count = 0;
    const hipError_t count_result = hipGetDeviceCount(&device_count);
    if (count_result != hipSuccess) {
        error = "No usable HIP device: hipGetDeviceCount failed: ";
        error += hipGetErrorString(count_result);
        error += ". Check ROCm access to /dev/kfd and /dev/dri.";
        return false;
    }
    if (device_count == 0) {
        error = "No usable HIP device: hipGetDeviceCount reported zero devices.";
        return false;
    }

    if (requested_index >= device_count) {
        error = "Requested HIP device " + std::to_string(requested_index)
                + " is unavailable; detected " + std::to_string(device_count) + " device(s).";
        return false;
    }

    if (requested_index >= 0) {
        if (!inspect_device(requested_index, device, error)) {
            return false;
        }
        if (!is_gfx906(device)) {
            error = "Selected GPU " + std::to_string(requested_index) + " (" + device.name
                    + ") reports architecture " + device.architecture
                    + "; MIInfer requires gfx906 and will not fall back.";
            return false;
        }
    } else {
        bool found = false;
        for (int index = 0; index < device_count; ++index) {
            DeviceInfo candidate;
            std::string candidate_error;
            if (inspect_device(index, candidate, candidate_error) && is_gfx906(candidate)) {
                device = candidate;
                found = true;
                break;
            }
        }
        if (!found) {
            error = "No gfx906 HIP device was detected among " + std::to_string(device_count)
                    + " device(s); MIInfer does not support another architecture.";
            return false;
        }
    }

    MIINFER_HIP_CHECK(hipSetDevice(device.index));
    return true;
}

void print_device_info(const DeviceInfo& device, std::ostream& output) {
    output << "GPU index: " << device.index << '\n'
           << "GPU name: " << device.name << '\n'
           << "Detected architecture: " << device.architecture << '\n'
           << "Total VRAM: " << device.total_vram_bytes << " bytes\n"
           << "MIInfer contract: gfx906 compatible\n";
}

}  // namespace miinfer
