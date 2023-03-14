/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Jan Ciesko (jciesko@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef RACERLIB_RDMA_WORKER
#define RACERLIB_RDMA_WORKER

#include <AccessCache.hpp>
#include <Helpers.hpp>

namespace Kokkos {
namespace Experimental {
namespace RACERlib {

template <typename T>
struct RdmaScatterGatherWorker {
  int my_rank;
  int num_ranks;

  static constexpr uint32_t queue_size = QUEUE_SIZE;

  KOKKOS_FUNCTION T get(int pe, uint32_t offset);

  KOKKOS_FUNCTION
  T request(int pe, uint32_t offset);

  Cache::RemoteCache cache;

  // Array of size num_ranks, a running count of element requests
  // generated by worker threads
  uint64_t *tx_element_request_ctrs;

  // Array of size num_ranks, a running count of element requests
  // completed and acked to worker threads. This is a mirror of
  // ack_ctrs_h that guarantees ack_ctrs_d[i] <= ack_cstr_h[i]
  // and that ack_ctrs_d will eventually equal ack_ctrs_h
  uint64_t *ack_ctrs_d;

  // Array of size num_ranks, a running count of element replies
  // back to each PE
  uint64_t *tx_element_reply_ctrs;

  // Array of size num_ranks, a running count of element requests
  // aggregated and processed by the request threads
  uint64_t *tx_element_aggregate_ctrs;

  // Array of size num_ranks*queue_length
  uint32_t *tx_element_request_queue;

  // Array of size num_ranks, the number of times we have wrapped around
  // the circular buffer queue for each PE
  uint32_t *tx_element_request_trip_counts;

  // Array of size num_ranks*queue_length, requests from remote PEs
  // are received here
  uint32_t *rx_element_request_queue;

  // Array of size num_ranks*queue_length, data is gathered here
  // and sent back to requesting PEs
  void *tx_element_reply_queue;

  // Array of size num_ranks*queue_length, gathered data from the remote PE
  // is received here
  void *rx_element_reply_queue;

  // Array of size num_ranks, a pointer that can be directly read
  // to access peer data, nullptr if no peer pointer exists
  void **direct_ptrs;

  // A running count of the number of block requests sent to all PEs
  uint64_t tx_block_request_ctr;

  // A running count of the number of block requests received from all PEs
  uint64_t rx_block_request_ctr;

  unsigned *request_done_flag;
  unsigned *response_done_flag;
  unsigned *fence_done_flag;
};

}  // namespace RACERlib
}  // namespace Experimental
}  // namespace Kokkos

#endif  // RACERLIB_RDMA_WORKER