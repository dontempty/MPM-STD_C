#pragma once

#include "common/types.hpp"
#include <cstddef>

namespace mpmstd::parallel::cuda_helpers {

// Device memory allocation.
// CUDA build : cudaMalloc.
// CPU build  : returns aligned host malloc (so callers can treat device_ptr() as a real pointer).
void* device_alloc(std::size_t bytes);
void  device_free (void* ptr);

// Bulk copies. Direction is implied by allocator semantics.
void copy_host_to_device(void* device_dst, const void* host_src, std::size_t bytes);
void copy_device_to_host(void* host_dst, const void* device_src, std::size_t bytes);
void copy_device_to_device(void* device_dst, const void* device_src, std::size_t bytes);

// Asynchronous variants take a generic stream handle (opaque on CPU build).
void copy_host_to_device_async(void* device_dst, const void* host_src, std::size_t bytes, void* stream);
void copy_device_to_host_async(void* host_dst, const void* device_src, std::size_t bytes, void* stream);

} // namespace mpmstd::parallel::cuda_helpers
