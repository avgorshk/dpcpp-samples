//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <CL/sycl.hpp>

#include <array>
#include <iostream>
#include <vector>
#include <set>
#include <string>

#include <CL/sycl/intel/fpga_extensions.hpp>

class MyGPUDeviceSelector : public cl::sycl::device_selector {
public:
  int operator()(const cl::sycl::device& device) const override {
    using namespace cl::sycl::info;

    const std::string name = device.get_info<device::name>();
    const std::string vendor = device.get_info<device::vendor>();
    if (vendor.empty()) {
      std::cout << "Host device" << ": " << name;
    } else {
      std::cout << vendor << ": " << name;
    }

    switch (device.get_info<device::device_type>()) {
      case device_type::host:
        std::cout << " (HOST)";
        break;
      case device_type::cpu:
        std::cout << " (CPU)";
        break;
      case device_type::gpu:
        std::cout << " (GPU)";
        break;
      case device_type::accelerator:
        std::cout << " (ACCELERATOR)";
        break;
      default:
        std::cout << " (UNKNOWN)";
        break;
    }
    std::cout << std::endl;

    return device.is_gpu();
  }
};

using namespace cl::sycl;

constexpr access::mode sycl_read = access::mode::read;
constexpr access::mode sycl_write = access::mode::write;

// This is the class used to name the kernel for the runtime.
// This must be done when the kernel is expressed as a lambda.
class VectorAdd;

// Note: Only way to be able to pass the arrays as params without 
// introducing magic numbers 
static const size_t ARRAY_SIZE = 10000; // 10,000

/**
 * Initialize the @param array of size ARRAY_SIZE with consecutive
 * elements from 0 to ARRAY_SIZE - 1
 */
void initialize_array(std::array<cl_int, ARRAY_SIZE>& arr);

/**
 * Computes the sum of two vectors in parallel using SYCL.
 */
void add_vectors_parallel(std::array<cl_int, ARRAY_SIZE>& sum_array,
  const std::array<cl_int, ARRAY_SIZE>& addend_array_1,
  const std::array<cl_int, ARRAY_SIZE>& addend_array_2);

/**
 * Computes the sum of two vectors in scalar with a simple loop.
 */
void add_vectors_scalar(std::array<cl_int, ARRAY_SIZE>& sum_array,
  const std::array<cl_int, ARRAY_SIZE>& addend_array_1,
  const std::array<cl_int, ARRAY_SIZE>& addend_array_2);

int main() {
  std::array<cl_int, ARRAY_SIZE> addend_array_1;
  std::array<cl_int, ARRAY_SIZE> addend_array_2;
  std::array<cl_int, ARRAY_SIZE> sum_array_parallel;
  std::array<cl_int, ARRAY_SIZE> sum_array_scalar;

  // Initialize vectors with values from 0 to ARRAY_SIZE - 1 
  initialize_array(addend_array_1);
  initialize_array(addend_array_2);
  initialize_array(sum_array_parallel);
  initialize_array(sum_array_scalar);

  // Add vectors in scalar and in parallel
  add_vectors_parallel(sum_array_parallel, addend_array_1, addend_array_2);
  add_vectors_scalar(sum_array_scalar, addend_array_1, addend_array_2);

  // Verify that the two sum vectors are equal 
  for (size_t i = 0; i < ARRAY_SIZE; i++) {
    if (sum_array_parallel[i] != sum_array_scalar[i]) {
      std::cout << "fail" << std::endl;
      return -1;
    }
  }
  std::cout << "success" << std::endl;
  return 0;
}

void initialize_array(std::array<cl_int, ARRAY_SIZE>& arr) {
  for (size_t i = 0; i < ARRAY_SIZE; i++) {
    arr[i] = i; // Initializing to the index 
  }
}

void add_vectors_parallel(std::array<cl_int, ARRAY_SIZE>& sum_array,
  const std::array<cl_int, ARRAY_SIZE>& addend_array_1,
  const std::array<cl_int, ARRAY_SIZE>& addend_array_2) {

  // SYCL model: A host is connected to OpenCL devices
  // Selectors are used to choose devices to be used
  // FPGA emulator and FPGA hardware devices can be targetted explicity
  // The default selector will choose the most performant device
  // Ex: It will use an accelerator if it can find one 

#ifdef INTEL_FPGA
  // FPGA device selector:  Emulator or Hardware
#ifdef FPGA_EMULATOR
  intel::fpga_emulator_selector device_selector;
#else
  intel::fpga_selector device_selector;
#endif
#else
    // Initializing the devices queue with the default selector
    // The device queue is used to enqueue the kernels and encapsulates
    // all the states needed for execution  
  MyGPUDeviceSelector device_selector;

#endif

  queue device_queue(device_selector);

  std::cout << std::endl;
  std::cout << "Device: "
    << device_queue.get_device().get_info<info::device::name>()
    << std::endl;

  // The size of amount of memory that will be given to the buffer 
  range<1> num_items{ ARRAY_SIZE };

  // Buffers are used to tell SYCL which data will be shared between the host
  // and the devices because they usually don't share physical memory
  // The pointer that's being passed as the first parameter transfers ownership
  // of the data to SYCL at runtime. The destructor is called when the buffer
  // goes out of scope and the data is given back to the std::arrays.
  // The second parameter specifies the size of the amount of memory being
  // given to the buffer.
  buffer<cl_int, 1> addend_1_buf(addend_array_1.data(), num_items);
  buffer<cl_int, 1> addend_2_buf(addend_array_2.data(), num_items);
  buffer<cl_int, 1> sum_buf(sum_array.data(), num_items);

  std::vector<size_t> group_id(num_items.size(), 0);
  buffer<size_t, 1> group_id_buf(group_id.data(), num_items);

  // queue::submit takes in a lambda that is passed in a command group handler
  // constructed at runtime. The lambda also contains a command group, which
  // contains the device-side operation and its dependencies 
  device_queue.submit([&](handler& cgh) {
    // Accessors are the only way to get access to the memory owned 
    // by the buffers initialized above. The first get_access template parameter 
    // specifies the access mode for the memory and the second template parameter 
    // is the type of memory to access the data from; this parameter has a default
    // value 
    auto addend_1_accessor = addend_1_buf.template get_access<sycl_read>(cgh);
    auto addend_2_accessor = addend_2_buf.template get_access<sycl_read>(cgh);

    // Note: Can use access::mode::discard_write instead of access::mode::write
    // because we're replacing the contents of the entire buffer.
    auto sum_accessor = sum_buf.template get_access<sycl_write>(cgh);

    auto group_accessor = group_id_buf.get_access<cl::sycl::access::mode::discard_write>(cgh);

    // Use parallel_for to run vector addition in parallel. This executes the
    // kernel. The first parameter is the number of work items to use and the 
    // second is the kernel, a lambda that specifies what to do per work item.
    // The template parameter VectorAdd is used to name the kernel at runtime.
    // The parameter passed to the lambda is the work item id of the current 
    // item.
    cgh.parallel_for<class VectorAdd>(num_items, [=](nd_item<1> wiID) {
      sum_accessor[wiID.get_global_id()] = addend_1_accessor[wiID.get_global_id()] +
        addend_2_accessor[wiID.get_global_id()];
      group_accessor[wiID.get_global_id()] = wiID.get_group(0);
    });
  });
  // SYCL will enqueue and run the kernel. Recall that the buffer's data is 
  // given back to the host at the end of the method's scope. 

  auto group_accessor = group_id_buf.get_access<cl::sycl::access::mode::read>();
  
  std::set<size_t> groups;
  for (size_t i = 0; i < num_items.size(); ++i) {
    groups.insert(group_accessor[i]);
  }

  std::cout << "Number of work-groups discovered: " << groups.size() << std::endl;
}

void add_vectors_scalar(std::array<cl_int, ARRAY_SIZE>& sum_array,
  const std::array<cl_int, ARRAY_SIZE>& addend_array_1,
  const std::array<cl_int, ARRAY_SIZE>& addend_array_2) {
  for (size_t i = 0; i < ARRAY_SIZE; i++) {
    sum_array[i] = addend_array_1[i] + addend_array_2[i];
  }
}
