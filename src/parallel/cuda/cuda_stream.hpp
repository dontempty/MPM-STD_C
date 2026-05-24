#pragma once

namespace mpmstd::parallel::cuda_helpers {

// Opaque stream handle. On CPU build it is always nullptr; passing it around
// keeps the API uniform between the two builds.
//
// Future expansion (M5'+): wrap as a class with a pool of cudaStream_t objects
// and allow round-robin assignment to overlap compute / halo exchange / IO.

class Stream {
public:
  Stream();
  ~Stream();

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  // Native handle (cudaStream_t cast to void*) or nullptr on CPU build.
  void* native() const { return native_; }

  void synchronize();

private:
  void* native_ = nullptr;
};

// Returns the default (legacy) stream wrapped in a Stream-shaped object.
Stream& default_stream();

} // namespace mpmstd::parallel::cuda_helpers
