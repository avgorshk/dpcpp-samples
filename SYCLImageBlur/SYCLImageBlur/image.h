#ifndef SYCLIMAGEBLUR_IMAGE_H
#define SYCLIMAGEBLUR_IMAGE_H

#include <algorithm>
#include <vector>

#include <assert.h>

#include <CL/sycl.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "external/stb_image_write.h"

namespace image {

struct Image {
  std::vector<float> r;
  std::vector<float> g;
  std::vector<float> b;
  int w;
  int h;
};

inline bool Load(const char* filename, Image& image) {
  assert(filename != nullptr);

  int comp = -1;
  uint8_t* data = stbi_load(filename, &image.w, &image.h, &comp, 0);
  if (data == nullptr) {
    return false;
  } else if (comp != 3) {
    stbi_image_free(data);
    return false;
  }

  image.r.resize(image.w * image.h);
  image.g.resize(image.w * image.h);
  image.b.resize(image.w * image.h);

  for (int i = 0; i < image.w * image.h; ++i) {
    image.r[i] = static_cast<float>(data[3 * i + 0]) / 255.0f;
    image.g[i] = static_cast<float>(data[3 * i + 1]) / 255.0f;
    image.b[i] = static_cast<float>(data[3 * i + 2]) / 255.0f;
  }

  stbi_image_free(data);
  return true;
}

inline bool SavePng(const char* filename, const Image& image) {
  assert(filename != nullptr);

  std::vector<uint8_t> data(3 * image.w * image.h, 0);
  for (int i = 0; i < image.w * image.h; ++i) {
    float r = image.r[i];
    float g = image.g[i];
    float b = image.b[i];

    float max = std::max(r, std::max(g, b));
    if (max > 1.0f) {
      r *= 1.0f / max;
      g *= 1.0f / max;
      b *= 1.0f / max;
    }
    data[3 * i + 0] = static_cast<uint8_t>(255.0f * std::max(
      0.0f, std::min(1.0f, r)));
    data[3 * i + 1] = static_cast<uint8_t>(255.0f * std::max(
      0.0f, std::min(1.0f, g)));
    data[3 * i + 2] = static_cast<uint8_t>(255.0f * std::max(
      0.0f, std::min(1.0f, b)));
  }

  int status = stbi_write_png(filename, image.w, image.h,
                              3 /*RGB*/, data.data(),
                              3 * image.w * sizeof(uint8_t));
  return status == 1 ? true : false;
}

} // namespace image

#endif