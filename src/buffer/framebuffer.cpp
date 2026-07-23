#include "buffer/framebuffer.hpp"

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "buffer/kernels/update_framebuffer.cuh"
#include "utils/cuda.hpp"

namespace fluidsim::buffer {

struct Framebuffer::Impl {
  cudaGraphicsResource_t resource;
};

Framebuffer::Framebuffer(u32 width, u32 height) : width_ {width}, height_ {height}, pimpl_ {std::make_unique<Impl>()} {
  glGenTextures(1, &texture_id_);
  glBindTexture(GL_TEXTURE_2D, texture_id_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width_, height_, 0, GL_RGBA, GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

  const auto flags {cudaGraphicsRegisterFlagsWriteDiscard | cudaGraphicsRegisterFlagsSurfaceLoadStore};
  utils::check_cuda_error(cudaGraphicsGLRegisterImage(&pimpl_->resource, texture_id_, GL_TEXTURE_2D, flags));
}

Framebuffer::~Framebuffer() {
  utils::check_cuda_error(cudaGraphicsUnregisterResource(pimpl_->resource));
  glDeleteTextures(1, &texture_id_);
}

auto Framebuffer::update(simulation::GridView grid) -> void {
  auto texture {cudaArray_t {}};
  auto surface {cudaSurfaceObject_t {}};
  auto descriptor {cudaResourceDesc {}};

  utils::check_cuda_error(cudaGraphicsMapResources(1, &pimpl_->resource));
  utils::check_cuda_error(cudaGraphicsSubResourceGetMappedArray(&texture, pimpl_->resource, 0, 0));

  descriptor.resType = cudaResourceTypeArray;
  descriptor.res.array.array = texture;
  utils::check_cuda_error(cudaCreateSurfaceObject(&surface, &descriptor));

  kernels::update_framebuffer(surface, grid);
  utils::check_async_cuda_error();

  utils::check_cuda_error(cudaDestroySurfaceObject(surface));
  utils::check_cuda_error(cudaGraphicsUnmapResources(1, &pimpl_->resource));
}

auto Framebuffer::texture_id() const -> GLuint {
  return texture_id_;
}

} // namespace fluidsim::buffer
