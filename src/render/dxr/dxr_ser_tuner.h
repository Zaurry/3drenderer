#pragma once
#include <array>
#include <cmath>
#include <cstdint>

namespace renderer {
// Each variant gets its own warmup and 24 completed GPU measurements. Tags
// follow the fence completion, not the CPU frame currently being recorded.
class DxrSerTuner {
public:
    void reset() {++generation_;warmup_.fill(8);count_.fill(0);sum_.fill(0);}
    bool complete() const {return count_[0]==samples && count_[1]==samples;}
    bool use_ser() const {return !complete()?count_[1]<samples:sum_[1]<sum_[0]*.98;}
    std::uint64_t tag(std::uint32_t frame,bool ser,bool eligible) const {
        return (generation_<<32)|(std::uint64_t(frame&0x3fffffff)<<2)|(eligible?2u:0u)|(ser?1u:0u);
    }
    void observe(std::uint64_t tag,float milliseconds) {
        if(tag==last_tag_)return;last_tag_=tag;
        if((tag>>32)!=generation_ || !(tag&2) || !std::isfinite(milliseconds) || milliseconds<=0)return;
        const unsigned variant=unsigned(tag&1);
        if(warmup_[variant]){--warmup_[variant];return;}
        if(count_[variant]<samples){sum_[variant]+=milliseconds;++count_[variant];}
    }
    float ser_ms() const {return count_[1]?float(sum_[1]/count_[1]):0;}
    float trace_ms() const {return count_[0]?float(sum_[0]/count_[0]):0;}
    float speedup() const {return complete()?trace_ms()/ser_ms()-1:0;}
private:
    static constexpr unsigned samples=24;
    std::uint64_t generation_=1,last_tag_=0;
    std::array<unsigned,2> warmup_{8,8},count_{};
    std::array<double,2> sum_{};
};
} // namespace renderer
