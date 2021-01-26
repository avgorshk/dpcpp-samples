// dpcpp -O2 -o bmul main.cc

#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

#define GROUP_SIZE 8

#define A 0.25f
#define B 0.125f

int main() {
  int n = 2048;
  std::vector<float> a(n * n, A);
  std::vector<float> b(n * n, B);
  std::vector<float> c(n * n, 0.0f);

  try {
    sycl::queue queue(sycl::gpu_selector{}, sycl::async_handler{});

    sycl::buffer<float, 1> buf_a(a.data(), a.size());
    sycl::buffer<float, 1> buf_b(b.data(), b.size());
    sycl::buffer<float, 1> buf_c(c.data(), c.size());

    queue.submit([&](auto& cgh) {
      auto dev_a = buf_a.get_access<sycl::access::mode::read>(cgh);
      auto dev_b = buf_b.get_access<sycl::access::mode::read>(cgh);
      auto dev_c = buf_c.get_access<sycl::access::mode::write>(cgh);

      using LocalAccessor =
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local>;
      LocalAccessor block_a(GROUP_SIZE * GROUP_SIZE, cgh);
      LocalAccessor block_b(GROUP_SIZE * GROUP_SIZE, cgh);

      cgh.parallel_for(
          sycl::nd_range<2>(sycl::range<2>(n, n), sycl::range<2>(GROUP_SIZE, GROUP_SIZE)),
          [=](sycl::nd_item<2> item) {
            int local_row = item.get_local_id(0);
            int local_col = item.get_local_id(1);
            int global_row = GROUP_SIZE * item.get_group(0) + local_row;
            int global_col = GROUP_SIZE * item.get_group(1) + local_col;
            int block_count = n / GROUP_SIZE;
            float sum = 0.0f;
            
            for (int i = 0; i < block_count; ++i) {
              // Read block from local memory
              block_a[local_row * GROUP_SIZE + local_col] = dev_a[global_row * n + (GROUP_SIZE * i + local_col)];
              block_b[local_row * GROUP_SIZE + local_col] = dev_b[(GROUP_SIZE * i + local_row) * n + global_col];
              item.barrier(sycl::access::fence_space::local_space);

              // Compute block
              for (int k = 0; k < GROUP_SIZE; ++k) {
                sum += block_a[local_row * GROUP_SIZE + k] * block_b[k * GROUP_SIZE + local_col];
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            dev_c[global_row * n + global_col] = sum;
          });
    });
    queue.wait_and_throw();
  } catch (...) {
    std::cout << "Something goes wrong" << std::endl;
    return 0;
  }

  for (size_t i = 0; i < c.size(); ++i) {
    if (c[i] != A * B * n) {
      std::cout << "Error" << std::endl;
      return 0;
    }
  }

  std::cout << "OK" << std::endl;
  return 0;
}