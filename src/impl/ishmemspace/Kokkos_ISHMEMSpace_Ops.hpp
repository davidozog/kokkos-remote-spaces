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

#ifndef KOKKOS_REMOTESPACES_ISHMEM_OPS_HPP
#define KOKKOS_REMOTESPACES_ISHMEM_OPS_HPP

#include <ishmem.h>
#include <type_traits>

namespace Kokkos {
namespace Impl {

#define KOKKOS_REMOTESPACES_P(type, op)                                       \
  static KOKKOS_INLINE_FUNCTION void shmem_type_p(type *ptr, const type &val, \
                                                  int pe) {                   \
    op(ptr, val, pe);                                                         \
  }

KOKKOS_REMOTESPACES_P(char, ishmem_char_p)
KOKKOS_REMOTESPACES_P(unsigned char, ishmem_uchar_p)
KOKKOS_REMOTESPACES_P(short, ishmem_short_p)
KOKKOS_REMOTESPACES_P(unsigned short, ishmem_ushort_p)
KOKKOS_REMOTESPACES_P(int, ishmem_int_p)
KOKKOS_REMOTESPACES_P(unsigned int, ishmem_uint_p)
KOKKOS_REMOTESPACES_P(long, ishmem_long_p)
KOKKOS_REMOTESPACES_P(unsigned long, ishmem_ulong_p)
KOKKOS_REMOTESPACES_P(long long, ishmem_longlong_p)
KOKKOS_REMOTESPACES_P(unsigned long long, ishmem_ulonglong_p)
KOKKOS_REMOTESPACES_P(float, ishmem_float_p)
KOKKOS_REMOTESPACES_P(double, ishmem_double_p)

#undef KOKKOS_REMOTESPACES_P

#define KOKKOS_REMOTESPACES_G(type, op)                                \
  static KOKKOS_INLINE_FUNCTION type shmem_type_g(type *ptr, int pe) { \
    return op(ptr, pe);                                                \
  }

KOKKOS_REMOTESPACES_G(char, ishmem_char_g)
KOKKOS_REMOTESPACES_G(unsigned char, ishmem_uchar_g)
KOKKOS_REMOTESPACES_G(short, ishmem_short_g)
KOKKOS_REMOTESPACES_G(unsigned short, ishmem_ushort_g)
KOKKOS_REMOTESPACES_G(int, ishmem_int_g)
KOKKOS_REMOTESPACES_G(unsigned int, ishmem_uint_g)
KOKKOS_REMOTESPACES_G(long, ishmem_long_g)
KOKKOS_REMOTESPACES_G(unsigned long, ishmem_ulong_g)
KOKKOS_REMOTESPACES_G(long long, ishmem_longlong_g)
KOKKOS_REMOTESPACES_G(unsigned long long, ishmem_ulonglong_g)
KOKKOS_REMOTESPACES_G(float, ishmem_float_g)
KOKKOS_REMOTESPACES_G(double, ishmem_double_g)

#undef KOKKOS_REMOTESPACES_G

#define KOKKOS_REMOTESPACES_ATOMIC_SET(type, op)            \
  static KOKKOS_INLINE_FUNCTION void shmem_type_atomic_set( \
      type *ptr, type value, int pe) {                      \
    return op(ptr, value, pe);                              \
  }

KOKKOS_REMOTESPACES_ATOMIC_SET(int, ishmem_int_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(unsigned int, ishmem_uint_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(long, ishmem_long_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(unsigned long, ishmem_ulong_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(long long, ishmem_longlong_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(unsigned long long, ishmem_ulonglong_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(float, ishmem_float_atomic_set)
KOKKOS_REMOTESPACES_ATOMIC_SET(double, ishmem_double_atomic_set)

#undef KOKKOS_REMOTESPACES_ATOMIC_SET

#define KOKKOS_REMOTESPACES_ATOMIC_FETCH(type, op)                      \
  static KOKKOS_INLINE_FUNCTION type shmem_type_atomic_fetch(type *ptr, \
                                                             int pe) {  \
    return op(ptr, pe);                                                 \
  }

KOKKOS_REMOTESPACES_ATOMIC_FETCH(int, ishmem_int_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(unsigned int, ishmem_uint_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(long, ishmem_long_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(unsigned long, ishmem_ulong_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(long long, ishmem_longlong_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(unsigned long long,
                                 ishmem_ulonglong_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(float, ishmem_float_atomic_fetch)
KOKKOS_REMOTESPACES_ATOMIC_FETCH(double, ishmem_double_atomic_fetch)

#undef KOKKOS_REMOTESPACES_ATOMIC_FETCH

#define KOKKOS_REMOTESPACES_ATOMIC_ADD(type, op)            \
  static KOKKOS_INLINE_FUNCTION void shmem_type_atomic_add( \
      type *ptr, type value, int pe) {                      \
    return op(ptr, value, pe);                              \
  }

KOKKOS_REMOTESPACES_ATOMIC_ADD(int, ishmem_int_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(unsigned int, ishmem_uint_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(long, ishmem_long_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(unsigned long, ishmem_ulong_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(long long, ishmem_longlong_atomic_add)
KOKKOS_REMOTESPACES_ATOMIC_ADD(unsigned long long, ishmem_ulonglong_atomic_add)

#undef KOKKOS_REMOTESPACES_ATOMIC_ADD

#define KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(type, op)            \
  static KOKKOS_INLINE_FUNCTION type shmem_type_atomic_fetch_add( \
      type *ptr, type value, int pe) {                            \
    return op(ptr, value, pe);                                    \
  }

KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(int, ishmem_int_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(unsigned int,
                                     ishmem_uint_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(long, ishmem_long_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(unsigned long,
                                     ishmem_ulong_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(long long,
                                     ishmem_longlong_atomic_fetch_add)
KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD(unsigned long long,
                                     ishmem_ulonglong_atomic_fetch_add)

#undef KOKKOS_REMOTESPACES_ATOMIC_FETCH_ADD

#define KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(type, op)            \
  static KOKKOS_INLINE_FUNCTION type shmem_type_atomic_compare_swap( \
      type *ptr, type cond, type value, int pe) {                    \
    return op(ptr, cond, value, pe);                                 \
  }
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(int, ishmem_int_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(unsigned int,
                                        ishmem_uint_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(long, ishmem_long_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(unsigned long,
                                        ishmem_ulong_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(long long,
                                        ishmem_longlong_atomic_compare_swap)
KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP(unsigned long long,
                                        ishmem_ulonglong_atomic_compare_swap)

#undef KOKKOS_REMOTESPACES_ATOMIC_COMPARE_SWAP

#define KOKKOS_REMOTESPACES_ATOMIC_SWAP(type, op)            \
  static KOKKOS_INLINE_FUNCTION type shmem_type_atomic_swap( \
      type *ptr, type value, int pe) {                       \
    return op(ptr, value, pe);                               \
  }
KOKKOS_REMOTESPACES_ATOMIC_SWAP(int, ishmem_int_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(unsigned int, ishmem_uint_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(long, ishmem_long_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(unsigned long, ishmem_ulong_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(long long, ishmem_longlong_atomic_swap)
KOKKOS_REMOTESPACES_ATOMIC_SWAP(unsigned long long,
                                ishmem_ulonglong_atomic_swap)

#undef KOKKOS_REMOTESPACES_ATOMIC_SWAP

template <class T, class Traits, typename Enable = void>
struct ISHMEMDataElement {};

// Atomic Operators
template <class T, class Traits>
struct ISHMEMDataElement<
    T, Traits,
    typename std::enable_if<Traits::memory_traits::is_atomic>::type> {
  typedef const T const_value_type;
  typedef T non_const_value_type;
  T *ptr;
  int pe;

  KOKKOS_INLINE_FUNCTION
  ISHMEMDataElement(T *ptr_, int pe_, int i_) : ptr(ptr_ + i_), pe(pe_) {}

  KOKKOS_INLINE_FUNCTION
  const_value_type operator=(const_value_type &val) const {
    shmem_type_atomic_set(ptr, val, pe);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  void inc() const {
    T tmp;
    tmp = 1;
    shmem_type_atomic_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  void dec() const {
    T tmp;
    tmp = 0 - 1;
    shmem_type_atomic_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++() const {
    T tmp;
    tmp = 1;
    return shmem_type_atomic_fetch_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--() const {
    T tmp;
    tmp = 0 - 1;
    return shmem_type_atomic_fetch_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++(int) const {
    T tmp;
    tmp = 1;
    return shmem_type_atomic_fetch_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--(int) const {
    T tmp;
    tmp = 0 - 1;
    return shmem_type_atomic_fetch_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+=(const_value_type &val) const {
    return shmem_type_atomic_fetch_add(ptr, val, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-=(const_value_type &val) const {
    T tmp;
    tmp = 0 - val;
    return shmem_type_atomic_fetch_add(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp * val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp / val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp % val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp & val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval, pe);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp ^ val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval, pe);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp | val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval, pe);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp << val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval, pe);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>=(const_value_type &val) const {
    T oldval, newval, tmp;
    oldval = shmem_type_g(ptr, pe);
    do {
      tmp    = oldval;
      newval = tmp >> val;
      oldval = shmem_type_atomic_compare_swap(ptr, tmp, newval, pe);
    } while (tmp != oldval);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp + val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp - val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp * val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp / val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp % val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator!() const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return !tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&&(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp && val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator||(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp || val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp & val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp | val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp ^ val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator~() const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return ~tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<(const unsigned int &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp << val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>(const unsigned int &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp >> val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp == val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp != val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp >= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp <= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp < val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp > val;
  }

  KOKKOS_INLINE_FUNCTION
  operator const_value_type() const {
    T tmp;
    tmp = shmem_type_atomic_fetch(ptr, pe);
    return tmp;
  }
};

// Default Operators
template <class T, class Traits>
struct ISHMEMDataElement<
    T, Traits,
    typename std::enable_if<!Traits::memory_traits::is_atomic>::type> {
  typedef const T const_value_type;
  typedef T non_const_value_type;
  T *ptr;
  int pe;

  KOKKOS_INLINE_FUNCTION
  ISHMEMDataElement(T *ptr_, int pe_, int i_) : ptr(ptr_ + i_), pe(pe_) {}

  KOKKOS_INLINE_FUNCTION
  const_value_type operator=(const_value_type &val) const {
    shmem_type_p(ptr, val, pe);
    return val;
  }

  KOKKOS_INLINE_FUNCTION
  void inc() const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp++;
    shmem_type_p(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  void dec() const {
    T tmp;
    shmem_type_g(ptr, pe);
    tmp--;
    shmem_type_p(ptr, tmp, pe);
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++() const {
    T tmp;
    shmem_type_g(ptr, pe);
    tmp++;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--() const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp--;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator++(int) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp++;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator--(int) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp--;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp += val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp -= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp *= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp /= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp %= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp &= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp ^= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp |= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp <<= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    tmp >>= val;
    shmem_type_p(ptr, tmp, pe);
    return tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator+(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp + val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator-(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp - val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator*(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp * val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator/(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp / val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator%(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp % val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator!() const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return !tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&&(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp && val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator||(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp || val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator&(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp & val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator|(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp | val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator^(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp ^ val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator~() const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return ~tmp;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator<<(const unsigned int &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp << val;
  }

  KOKKOS_INLINE_FUNCTION
  const_value_type operator>>(const unsigned int &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp >> val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator==(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp == val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator!=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp != val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp >= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<=(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp <= val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator<(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp < val;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator>(const_value_type &val) const {
    T tmp;
    tmp = shmem_type_g(ptr, pe);
    return tmp > val;
  }

  KOKKOS_INLINE_FUNCTION
  operator const_value_type() const {
    T tmp = shmem_type_g((double *)ptr, pe);
    return tmp;
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_ISHMEM_OPS_HPP
