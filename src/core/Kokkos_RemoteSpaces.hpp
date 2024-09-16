//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
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

#ifndef KOKKOS_REMOTESPACES_HPP
#define KOKKOS_REMOTESPACES_HPP
#include <Kokkos_Core.hpp>

#include <iostream>

namespace Kokkos {
namespace Experimental {

namespace Impl {
class InternalSpecializeTag {};
}  // namespace Impl

template <class T = Kokkos::Experimental::Impl::InternalSpecializeTag>
class RemoteSpacesSpecializeTag {};

using RemoteSpaceSpecializeTag = RemoteSpacesSpecializeTag<>;

enum RemoteSpaces_MemoryAllocationMode : int { Symmetric, Cached };
}  // namespace Experimental
}  // namespace Kokkos

#ifdef KRS_ENABLE_SHMEMSPACE
namespace Kokkos {
namespace Experimental {
class SHMEMSpace;
}
}  // namespace Kokkos
#include <Kokkos_SHMEMSpace.hpp>
#endif

#ifdef KRS_ENABLE_NVSHMEMSPACE
namespace Kokkos {
namespace Experimental {
class NVSHMEMSpace;
}
}  // namespace Kokkos
#include <Kokkos_NVSHMEMSpace.hpp>
#endif

#ifdef KRS_ENABLE_ROCSHMEMSPACE
namespace Kokkos {
namespace Experimental {
class ROCSHMEMSpace;
}
}  // namespace Kokkos
#include <Kokkos_ROCSHMEMSpace.hpp>
#endif

#ifdef KRS_ENABLE_MPISPACE
namespace Kokkos {
namespace Experimental {
class MPISpace;
}
}  // namespace Kokkos
#include <Kokkos_MPISpace.hpp>
#endif

#ifdef KRS_ENABLE_ISHMEMSPACE
namespace Kokkos {
namespace Experimental {
class ISHMEMSpace;
}
}  // namespace Kokkos
#include <Kokkos_ISHMEMSpace.hpp>
#endif

namespace Kokkos {
namespace Experimental {

#ifdef KRS_ENABLE_NVSHMEMSPACE
typedef NVSHMEMSpace DefaultRemoteMemorySpace;
#else
#ifdef KRS_ENABLE_ROCSHMEMSPACE
typedef ROCSHMEMSpace DefaultRemoteMemorySpace;
#else
#ifdef KRS_ENABLE_SHMEMSPACE
typedef SHMEMSpace DefaultRemoteMemorySpace;
#else
#ifdef KRS_ENABLE_MPISPACE
typedef MPISpace DefaultRemoteMemorySpace;
#else
#ifdef KRS_ENABLE_ISHMEMSPACE
typedef ISHMEMSpace DefaultRemoteMemorySpace;
#else
error "At least one remote space must be selected."
#endif
#endif
#endif
#endif
#endif
}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_RESMOTESPACES_HPP
