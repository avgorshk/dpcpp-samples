#include <CL/sycl.hpp>

#include <chrono>
#include <iostream>
#include <memory>

#include <assert.h>
#include <stdint.h>

#include "image.h"

class ComputeRChannel;
class ComputeGChannel;
class ComputeBChannel;

using string = std::string;

struct ComputeChannel {
  using GlobalReadAccessor = cl::sycl::accessor<float, 1,
    cl::sycl::access::mode::read,
    cl::sycl::access::target::global_buffer>;
  using GlobalWriteAccessor = cl::sycl::accessor<float, 1,
    cl::sycl::access::mode::write,
    cl::sycl::access::target::global_buffer>;
  using LocalAccessor = cl::sycl::accessor<float, 1,
    cl::sycl::access::mode::read_write,
    cl::sycl::access::target::local>;

  GlobalReadAccessor in;
  GlobalWriteAccessor out;
  GlobalReadAccessor filter;
  LocalAccessor buffer;

  int filter_size;
  int width;
  int height;

  void operator()(cl::sycl::nd_item<2> item) {
    size_t lx = item.get_local_id(0);
    size_t ly = item.get_local_id(1);
    size_t gx = item.get_global_id(0);
    size_t gy = item.get_global_id(1);

    gx += filter_size / 2;
    gy += filter_size / 2;

    buffer[ly * filter_size + lx] = filter[ly * filter_size + lx];
    item.barrier(cl::sycl::access::fence_space::local_space);

    // Internal pixels

    if (gx >= filter_size / 2 && gx < width - filter_size / 2 &&
        gy >= filter_size / 2 && gy < height - filter_size / 2) {
      float sum = 0.0;
      for (int y = 0; y < filter_size; ++y) {
        int idy = gy - filter_size / 2 + y;
        for (int x = 0; x < filter_size; ++x) {
          int idx = gx - filter_size / 2 + x;
          sum += in[idy * width + idx] * buffer[y * filter_size + x];
        }
      }

      out[gy * width + gx] = sum;
    }

    // Boundary pixels

    size_t gidx = item.get_global_id(0);
    size_t gidy = item.get_global_id(1);
    size_t gsize = item.get_global_range(0);

    for (int gy = gidy; gy < filter_size / 2; gy += gsize) {
      for (int gx = gidx; gx < width; gx += gsize) {
        out[gy * width + gx] = 0;

        for (int y = cl::sycl::max(0, filter_size / 2 - gy);
             y < cl::sycl::min(filter_size,
                               height - gy + filter_size / 2);
             ++y) {
          int idy = gy - filter_size / 2 + y;
          for (int x = cl::sycl::max(0, filter_size / 2 - gx);
                x < cl::sycl::min(filter_size,
                                  width - gx + filter_size / 2);
               ++x) {
            int idx = gx - filter_size / 2 + x;
            out[gy * width + gx] += in[idy * width + idx] *
              buffer[y * filter_size + x];
          }
        }
      }
    }

    for (int gy = gidy + filter_size / 2;
         gy < height - filter_size / 2; gy += gsize) {
      for (int gx = gidx; gx < filter_size / 2; gx += gsize) {
        out[gy * width + gx] = 0;

        for (int y = cl::sycl::max(0, filter_size / 2 - gy);
             y < cl::sycl::min(filter_size,
                               height - gy + filter_size / 2);
             ++y) {
          int idy = gy - filter_size / 2 + y;
          for (int x = cl::sycl::max(0, filter_size / 2 - gx);
               x < cl::sycl::min(filter_size,
                                 width - gx + filter_size / 2);
               ++x) {
            int idx = gx - filter_size / 2 + x;
            out[gy * width + gx] += in[idy * width + idx] *
              buffer[y * filter_size + x];
          }
        }
      }

      for (int gx = gidx + width - filter_size / 2;
           gx < width; gx += gsize) {
        out[gy * width + gx] = 0;

        for (int y = cl::sycl::max(0, filter_size / 2 - gy);
             y < cl::sycl::min(filter_size,
                               height - gy + filter_size / 2);
             ++y) {
          int idy = gy - filter_size / 2 + y;
          for (int x = cl::sycl::max(0, filter_size / 2 - gx);
               x < cl::sycl::min(filter_size,
                                 width - gx + filter_size / 2);
               ++x) {
            int idx = gx - filter_size / 2 + x;
            out[gy * width + gx] += in[idy * width + idx] *
              buffer[y * filter_size + x];
          }
        }
      }
    }

    for (int gy = gidy + height - filter_size / 2;
         gy < height; gy += gsize) {
      for (int gx = gidx; gx < width; gx += gsize) {
        out[gy * width + gx] = 0;

        for (int y = cl::sycl::max(0, filter_size / 2 - gy);
             y < cl::sycl::min(filter_size,
                               height - gy + filter_size / 2);
             ++y) {
          int idy = gy - filter_size / 2 + y;
          for (int x = cl::sycl::max(0, filter_size / 2 - gx);
               x < cl::sycl::min(filter_size,
                                 width - gx + filter_size / 2);
               ++x) {
            int idx = gx - filter_size / 2 + x;
            out[gy * width + gx] += in[idy * width + idx] *
              buffer[y * filter_size + x];
          }
        }
      }
    }
  }
};

void GetBlurFilter(int filter_size, std::vector<float>& filter) {
  filter.resize(filter_size * filter_size);
  for (int i = 0; i < filter_size * filter_size; ++i) {
    filter[i] = 1.0f / (filter_size * filter_size);
  }
}

uint64_t PerformBlur(const image::Image& input,
                  int filter_size,
                  image::Image& output,
                  cl::sycl::info::device_type device_type) {
  uint64_t exec_time = 0;
  
  output.w = input.w;
  output.h = input.h;
  output.r.resize(output.w * output.h);
  output.g.resize(output.w * output.h);
  output.b.resize(output.w * output.h);

  size_t gsx = input.w - 2 * (filter_size / 2);
  gsx = ((gsx + filter_size - 1) / filter_size) * filter_size;
  size_t gsy = input.h - 2 * (filter_size / 2);
  gsy = ((gsy + filter_size - 1) / filter_size) * filter_size;

  cl::sycl::range<2> global_size(gsx, gsy);
  cl::sycl::range<2> local_size(filter_size, filter_size);

  std::vector<float> filter;
  GetBlurFilter(filter_size, filter);
  assert(filter.size() == filter_size * filter_size);

  cl::sycl::buffer<float, 1> filter_buffer(filter.data(), filter.size());
  
  cl::sycl::buffer<float, 1> input_r_buffer(input.r.data(), input.r.size());
  cl::sycl::buffer<float, 1> input_g_buffer(input.g.data(), input.g.size());
  cl::sycl::buffer<float, 1> input_b_buffer(input.b.data(), input.b.size());

  cl::sycl::buffer<float, 1> output_r_buffer(output.r.data(), output.r.size());
  cl::sycl::buffer<float, 1> output_g_buffer(output.g.data(), output.g.size());
  cl::sycl::buffer<float, 1> output_b_buffer(output.b.data(), output.b.size());

  std::unique_ptr<cl::sycl::device_selector> selector(nullptr);
  if (device_type == cl::sycl::info::device_type::cpu) {
    selector.reset(new cl::sycl::cpu_selector);
  } else if (device_type == cl::sycl::info::device_type::gpu) {
    selector.reset(new cl::sycl::gpu_selector);
  } else if (device_type == cl::sycl::info::device_type::host) {
    selector.reset(new cl::sycl::host_selector);
  } else {
    return 0.0f;
  }
  
  try {
    cl::sycl::queue queue(*selector.get(), cl::sycl::async_handler{});
    std::cout << "Target device: " <<
      queue.get_info<cl::sycl::info::queue::device>().get_info<
        cl::sycl::info::device::name>() << std::endl;

    auto start = std::chrono::steady_clock::now();

    // Compute R channel
    queue.submit([&](cl::sycl::handler& cgh) {
      auto filter =
        filter_buffer.get_access<cl::sycl::access::mode::read>(cgh);
      auto in =
        input_r_buffer.get_access<cl::sycl::access::mode::read>(cgh);
      auto out =
        output_r_buffer.get_access<cl::sycl::access::mode::write>(cgh);

      cl::sycl::accessor<float, 1,
        cl::sycl::access::mode::read_write,
        cl::sycl::access::target::local>
        buffer(cl::sycl::range<1>(filter_size * filter_size), cgh);

      cgh.parallel_for(cl::sycl::nd_range<2>(global_size, local_size),
                        ComputeChannel{in, out, filter, buffer,
                                      filter_size, input.w, input.h});
    });

    // Compute G channel
    queue.submit([&](cl::sycl::handler& cgh) {
      auto filter =
        filter_buffer.get_access<cl::sycl::access::mode::read>(cgh);
      auto in =
        input_g_buffer.get_access<cl::sycl::access::mode::read>(cgh);
      auto out =
        output_g_buffer.get_access<cl::sycl::access::mode::write>(cgh);

      cl::sycl::accessor<float, 1,
        cl::sycl::access::mode::read_write,
        cl::sycl::access::target::local>
        buffer(cl::sycl::range<1>(filter_size * filter_size), cgh);

      cgh.parallel_for(cl::sycl::nd_range<2>(global_size, local_size),
                        ComputeChannel{ in, out, filter, buffer,
                                      filter_size, input.w, input.h });
    });

    // Compute B channel
    queue.submit([&](cl::sycl::handler& cgh) {
      auto filter =
        filter_buffer.get_access<cl::sycl::access::mode::read>(cgh);
      auto in =
        input_b_buffer.get_access<cl::sycl::access::mode::read>(cgh);
      auto out =
        output_b_buffer.get_access<cl::sycl::access::mode::write>(cgh);

      cl::sycl::accessor<float, 1,
        cl::sycl::access::mode::read_write,
        cl::sycl::access::target::local>
        buffer(cl::sycl::range<1>(filter_size * filter_size), cgh);

      cgh.parallel_for(cl::sycl::nd_range<2>(global_size, local_size),
                        ComputeChannel{ in, out, filter, buffer,
                                      filter_size, input.w, input.h });
    });
   
    queue.wait_and_throw();

    auto end = std::chrono::steady_clock::now();
    exec_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();
  } catch (std::exception e) {
    std::cout << "Error: " << e.what() << std::endl;
  }

  return exec_time;
}

void Usage() {
  std::cout << "Usage: SYCLImageBlur.exe <image.jpg> <filter_size> <cpu|gpu>" <<
    std::endl;
}

string GetBluredImageName(const string& image_name) {
  size_t dot_pos = image_name.rfind(".");
  if (dot_pos == string::npos) {
    return string();
  }
  return image_name.substr(0, dot_pos) + ".png";
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    Usage();
    return 0;
  }

  char* image_name = argv[1];
  int filter_size = atoi(argv[2]);
  assert(filter_size > 0);
  char* device = argv[3];

  std::cout << "Target image: " << image_name << std::endl;
  std::cout << "Target filter size: " << filter_size << std::endl;

  image::Image image;
  bool succeed = image::Load(image_name, image);
  if (!succeed) {
    std::cout << "Image " << image_name << " not found!" << std::endl;
    return 0;
  } else {
    std::cout << "Image " << image_name <<
      " is loaded successfully" << std::endl;
  }

  cl::sycl::info::device_type device_type =
    cl::sycl::info::device_type::cpu;
  if (strcmp(device, "gpu") == 0) {
    device_type = cl::sycl::info::device_type::gpu;
    if (filter_size > 16) {
      std::cout << "Filter size on GPU is limited to 16" << std::endl;
      return 0;
    }
  }

  image::Image blured_image;
  auto exec_time = PerformBlur(image, filter_size, blured_image, device_type);
  assert(exec_time > 0);

  string blured_image_name = GetBluredImageName(image_name);
  assert(!blured_image_name.empty());

  succeed = image::SavePng(blured_image_name.c_str(), blured_image);
  assert(succeed);
  std::cout << "Result image is stored to " << blured_image_name << std::endl;
  std::cout << "Execution time: " << exec_time << " ms" << std::endl;

  return 0;
}
