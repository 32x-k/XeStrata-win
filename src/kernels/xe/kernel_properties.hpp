// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

// SYCL required sizes are metadata; AMD register allocation reads this function
// attribute instead. Derive it from the actual kernel launch, only on that target.
// Keep the required WG/SG properties on the generated entry on every target.
#include <sycl/sycl.hpp>
namespace strata::kernels::xe::kernel_properties {
namespace ex=sycl::ext::oneapi::experimental;
struct FlatKey:ex::detail::compile_time_property_key<ex::detail::PropKind::MaxLinearWorkGroupSize> {
 template<size_t N> using value_t=ex::property_value<FlatKey,std::integral_constant<size_t,N>>;
};
}
namespace sycl::ext::oneapi::experimental::detail {
template<size_t N> struct HasCompileTimeEffect<strata::kernels::xe::kernel_properties::FlatKey::value_t<N>>:std::true_type {};
template<size_t N> struct PropertyMetaInfo<strata::kernels::xe::kernel_properties::FlatKey::value_t<N>> {
 static constexpr const char* name="amdgpu-flat-work-group-size";
 static constexpr const char* value=SizeListToStr<1,N>::value;
};
}
namespace strata::kernels::xe::kernel_properties {
template<size_t N,int SG,class F> struct Body {
 F function;
 auto get(ex::properties_tag) const {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
  return ex::properties{FlatKey::value_t<N>{},ex::work_group_size<N>,ex::sub_group_size<SG>};
#else
  return ex::properties{ex::work_group_size<N>,ex::sub_group_size<SG>};
#endif
 }
 [[gnu::always_inline]] void operator()(sycl::nd_item<1> item) const {[[clang::always_inline]] function(item);}
};
template<size_t N,int SG,class F> auto wrap(F function) {return Body<N,SG,F>{function};}
}
