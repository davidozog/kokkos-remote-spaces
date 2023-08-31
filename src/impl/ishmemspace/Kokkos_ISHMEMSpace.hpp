//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2024) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Contact: Jan Ciesko (jciesko@sandia.gov)
//
//@HEADER

#ifndef KOKKOS_ISHMEMSPACE_HPP
#define KOKKOS_ISHMEMSPACE_HPP

#include <cstring>
#include <iosfwd>
#include <string>
#include <typeinfo>

#include <Kokkos_Core.hpp>

#include <Kokkos_RemoteSpaces.hpp>
#include <mpi.h>
#include <ishmem.h>
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

class ISHMEMSpace {
 public:
#if defined(KOKKOS_ENABLE_SYCL)
  using execution_space = Kokkos::Experimental::SYCL;
#else
#error \
    "At least the following device execution space must be defined: Kokkos::Sycl."
#endif
  using memory_space = ISHMEMSpace;
  using device_type  = Kokkos::Device<execution_space, memory_space>;
  using size_type    = size_t;

  ISHMEMSpace();
  ISHMEMSpace(ISHMEMSpace &&rhs)      = default;
  ISHMEMSpace(const ISHMEMSpace &rhs) = default;
  ISHMEMSpace &operator=(ISHMEMSpace &&) = default;
  ISHMEMSpace &operator=(const ISHMEMSpace &) = default;
  ~ISHMEMSpace()                               = default;

  explicit ISHMEMSpace(const MPI_Comm &);

  void *allocate(const size_t arg_alloc_size) const;

  void deallocate(void *const arg_alloc_ptr, const size_t arg_alloc_size) const;

  void *allocate(const int *gids, const int &arg_local_alloc_size) const;

  void deallocate(const int *gids, void *const arg_alloc_ptr,
                  const size_t arg_alloc_size) const;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char *name() { return m_name; }

  static void fence();

  int allocation_mode;
  int64_t extent;

  void impl_set_allocation_mode(const int);
  void impl_set_extent(int64_t N);

 private:
  static constexpr const char *m_name = "ISHMEM";
  friend class Kokkos::Impl::SharedAllocationRecord<
      Kokkos::Experimental::ISHMEMSpace, void>;
};

KOKKOS_FUNCTION
int get_num_pes();
KOKKOS_FUNCTION
int get_my_pe();
KOKKOS_FUNCTION
size_t get_indexing_block_size(size_t size);
std::pair<size_t, size_t> getRange(size_t size, size_t pe);

}  // namespace Experimental
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <>
struct DeepCopy<HostSpace, Kokkos::Experimental::ISHMEMSpace> {
  DeepCopy(void *dst, const void *src, size_t);
};

template <>
struct DeepCopy<Kokkos::Experimental::ISHMEMSpace, HostSpace> {
  DeepCopy(void *dst, const void *src, size_t);
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::ISHMEMSpace,
                Kokkos::Experimental::ISHMEMSpace, ExecutionSpace> {
  DeepCopy(void *dst, const void *src, size_t n);
  DeepCopy(const ExecutionSpace &exec, void *dst, const void *src, size_t n);
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::ISHMEMSpace,
                         Kokkos::Experimental::ISHMEMSpace> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = false };
};

template <>
struct MemorySpaceAccess<Kokkos::HostSpace,
                         Kokkos::Experimental::ISHMEMSpace> {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::SYCLDeviceUSMSpace,
                         Kokkos::Experimental::ISHMEMSpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

}  // namespace Impl
}  // namespace Kokkos

#include <Kokkos_RemoteSpaces_Error.hpp>
#include <Kokkos_RemoteSpaces_Options.hpp>
#include <Kokkos_ISHMEMSpace_ViewTraits.hpp>
#include <Kokkos_RemoteSpaces_ViewLayout.hpp>
#include <Kokkos_RemoteSpaces_Helpers.hpp>
#include <Kokkos_RemoteSpaces_DeepCopy.hpp>
#include <Kokkos_RemoteSpaces_ViewOffset.hpp>
#include <Kokkos_ISHMEMSpace_Ops.hpp>
#include <Kokkos_ISHMEMSpace_BlockOps.hpp>
#include <Kokkos_RemoteSpaces_ViewMapping.hpp>
#include <Kokkos_ISHMEMSpace_AllocationRecord.hpp>
#include <Kokkos_ISHMEMSpace_DataHandle.hpp>
#include <Kokkos_RemoteSpaces_LocalDeepCopy.hpp>

#endif  // #define KOKKOS_ISHMEMSPACE_HPP
