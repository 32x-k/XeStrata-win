// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#include "strata/core/device.hpp"
#include "strata/core/runtime.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/gpu.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace gpu = strata::gpu;
namespace mmq = strata::prefill::mmq;
using Clock = std::chrono::steady_clock;
constexpr int K=4096, E=16, T=12;
void check(bool ok) { if (!ok) throw std::runtime_error(gpu::last_error()); }
void* alloc(size_t n) { void* p=gpu::alloc_device(n); check(p != nullptr); return p; }
template<class F> double measure(F f) {
    std::vector<double> v;
    for(int i=0;i<9;i++) { check(gpu::device_sync()); auto a=Clock::now(); f(); check(gpu::device_sync()); auto b=Clock::now(); if(i>=2) v.push_back(std::chrono::duration<double,std::milli>(b-a).count()); }
    std::sort(v.begin(),v.end()); return v[3];
}
struct Work {
    int d,m,n; bool direct_io; size_t gu,blob; void *weights,*xq,*hq; float *x,*g,*out; int32_t *bounds,*ids; const uint8_t** tab; mmq::Context ctx;
    Work(int dev,int rows,int ff,const void* host,bool direct):d(dev),m(rows),n(ff),direct_io(direct) {
        strata::core::OnDevice on(d); gu=mmq::matrix_bytes(T,2*n,K); blob=gu+mmq::matrix_bytes(T,K,n);
        weights=direct?nullptr:alloc(E*blob); if(weights) check(gpu::copy(weights,host,E*blob));
        if (direct_io) {
            check(posix_memalign((void**)&x,4096,size_t(m)*K*4)==0);
            check(posix_memalign((void**)&out,4096,size_t(m)*K*4)==0);
            check(gpu::host_register_on(d,x,size_t(m)*K*4)); check(gpu::host_register_on(d,out,size_t(m)*K*4));
        } else { x=(float*)alloc(size_t(m)*K*4); out=(float*)alloc(size_t(m)*K*4); }
        g=(float*)alloc(size_t(m)*2*n*4);
        xq=alloc(mmq::q8_bytes(m,K)); hq=alloc(mmq::q8_bytes(m,n)); bounds=(int32_t*)alloc((E+1)*4); ids=(int32_t*)alloc(m*4); tab=(const uint8_t**)alloc(E*sizeof(*tab));
        std::vector<float> hx(size_t(m)*K); for(size_t i=0;i<hx.size();i++) hx[i]=(int(i%251)-125)/250.f;
        std::vector<int32_t> hb(E+1); for(int e=0;e<=E;e++) hb[e]=e*(m/E);
        std::vector<const uint8_t*> ht(E); for(int e=0;e<E;e++) ht[e]=(const uint8_t*)(direct?host:weights)+e*blob;
        check(gpu::copy(x,hx.data(),hx.size()*4)); check(gpu::copy(bounds,hb.data(),hb.size()*4)); check(gpu::copy((void*)tab,ht.data(),ht.size()*sizeof(ht[0]))); mmq::iota(ids,m,nullptr); check(gpu::device_sync());
    }
    ~Work() { strata::core::OnDevice on(d); if (direct_io) { gpu::host_unregister_on(d,x); gpu::host_unregister_on(d,out); std::free(x); std::free(out); } else { gpu::free(x); gpu::free(out); } for(const void* p:{weights,(void*)g,xq,hq,(void*)bounds,(void*)ids,(void*)tab}) gpu::free(p); }
    void quant(){mmq::quantize(x,nullptr,xq,T,K,K,m,nullptr);}
    void gate(){ mmq::Product p; p.type=T;p.w_rows=2*n;p.w_cols=K;p.n=E;p.xq=xq;p.bounds=bounds;p.ids=ids;p.total_rows=m;p.max_rows=m/E;p.rows=m;p.dst=g;p.ld_dst=2*n;p.wtab=tab;ctx.run(p,nullptr); }
    void swiglu(){mmq::swiglu_quantize(g,hq,T,m,n,false,nullptr);}
    void down(){ mmq::Product p; p.type=T;p.w_rows=K;p.w_cols=n;p.n=E;p.xq=hq;p.bounds=bounds;p.ids=ids;p.total_rows=m;p.max_rows=m/E;p.rows=m;p.dst=out;p.ld_dst=K;p.wtab=tab;p.woff=gu;ctx.run(p,nullptr); }
    void all(){quant();gate();swiglu();down();}
    std::vector<float> result(){ std::vector<float> v(size_t(m)*K); check(gpu::copy(v.data(),out,v.size()*4)); return v; }
};
int main(){try {
    if(strata::core::Runtime::count()<2 || !mmq::supported(T)) throw std::runtime_error("two devices and Q4_K support required");
    for(int d=0;d<2;d++) std::printf("device %d: %s\n",d,strata::core::device_info(d).name.c_str());
    for(int n:{1280,640}) {
        if (n % 256 != 0) { std::printf("skip N_ff=%d: Q4_K down matrix requires n_ff divisible by 256\n", n); continue; }
        size_t gu=mmq::matrix_bytes(T,2*n,K), blob=gu+mmq::matrix_bytes(T,K,n), bytes=E*blob;
        void* host=nullptr; bool reg0=false,reg1=false, registered=false;
        if(posix_memalign(&host,4096,bytes)==0) { reg0=gpu::host_register_on(0,host,bytes); if(reg0) reg1=gpu::host_register_on(1,host,bytes); if(reg1) {strata::core::OnDevice on(1); registered=gpu::kernels_read_registered();} }
        if(!registered) { if(reg1) gpu::host_unregister_on(1,host); if(reg0) gpu::host_unregister_on(0,host); std::free(host); strata::core::OnDevice on(1); host=gpu::alloc_host(bytes); check(host && gpu::kernels_read_host(host)); }
        std::printf("N_ff=%d host=%s bytes=%zu\n",n,registered?"registered":"device-1 host USM",bytes);
        // Q4_K: two fp16 scales, 12 packed scale bytes, 128 packed quants.
        auto *h=(uint8_t*)host; for(size_t i=0;i<bytes;i+=144) { h[i]=0;h[i+1]=0x2c;h[i+2]=h[i+3]=0; std::memset(h+i+4,1,12); std::memset(h+i+16,0x21,128); }
        std::printf("device M quant_in gate_up swiglu_q down swiglu_pct total_ms\n");
        std::printf("crossover: M resident_ms stream_ms h2d_ms GB/s direct_ms max_abs_stream max_abs_direct\n");
        for(int m:{32,128,512,2048,4096}) {
            Work a(0,m,n,host,false), b(1,m,n,host,false), c(1,m,n,host,true); double total[2];
            for(int d=0;d<2;d++){ strata::core::OnDevice on(d); Work& w=d?b:a; double q=measure([&]{w.quant();}); double g=measure([&]{w.gate();}); double s=measure([&]{w.swiglu();}); double down=measure([&]{w.down();}); total[d]=measure([&]{w.all();}); std::printf("%d %d %.3f %.3f %.3f %.3f %.2f%% %.3f\n",d,m,q,g,s,down,100*s/(g+s+down),total[d]); std::fflush(stdout); }
            double direct_ms; { strata::core::OnDevice other(1); direct_ms=measure([&]{c.all();}); }
            strata::core::OnDevice on(0); auto ref=a.result(); double h2d=measure([&]{check(gpu::copy_async(a.weights,host,bytes,nullptr));}); double stream=measure([&]{check(gpu::copy_async(a.weights,host,bytes,nullptr)); a.all();}); auto streamed=a.result(); std::vector<float> direct; {strata::core::OnDevice other(1); direct=c.result();}
            double e1=0,e2=0; for(size_t i=0;i<ref.size();i++){if(!std::isfinite(ref[i])||!std::isfinite(streamed[i])||!std::isfinite(direct[i])) throw std::runtime_error("non-finite output"); e1=std::max(e1,double(std::abs(ref[i]-streamed[i]))); e2=std::max(e2,double(std::abs(ref[i]-direct[i])));}
            std::printf("cross %d %.3f %.3f %.3f %.2f %.3f %.8g %.8g\n",m,total[0],stream,h2d,bytes/(h2d*1e6),direct_ms,e1,e2); std::fflush(stdout); if(e1>1e-2||e2>1e-2) throw std::runtime_error("parity failed");
        }
        if(registered){gpu::host_unregister_on(1,host);gpu::host_unregister_on(0,host);std::free(host);} else {strata::core::OnDevice on(1);gpu::free(host);}
    }
} catch(const std::exception& e){std::fprintf(stderr,"benchmark failed: %s\n",e.what());return 1;} }
