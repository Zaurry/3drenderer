#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace renderer {
// Walker alias distribution. Keep the individual PMF: subtracting adjacent
// float CDF entries loses dim texels next to a bright HDR source.
struct DxrAliasEntry {
    float threshold=1,pmf=0;
    std::uint32_t alias=0,reserved=0;
};
static_assert(sizeof(DxrAliasEntry)==16);

inline std::vector<DxrAliasEntry> make_dxr_alias_table(std::span<const float> weights) {
    if(weights.size()>std::numeric_limits<std::uint32_t>::max())throw std::length_error("DXR alias table too large");
    const auto count=static_cast<std::uint32_t>(weights.size());
    std::vector<DxrAliasEntry> table(count);if(!count)return table;
    double total=0;
    for(float weight:weights) {
        if(!std::isfinite(weight) || weight<0)throw std::invalid_argument("Invalid DXR alias weight");
        total+=weight;
    }
    std::vector<double> scaled(count);
    std::vector<std::uint32_t> below_one,above_one;below_one.reserve(count);above_one.reserve(count);
    for(std::uint32_t i=0;i<count;++i) {
        const double pmf=total>0?weights[i]/total:1./count;
        table[i].pmf=static_cast<float>(pmf);table[i].alias=i;scaled[i]=pmf*count;
        (scaled[i]<1?below_one:above_one).push_back(i);
    }
    while(!below_one.empty() && !above_one.empty()) {
        const auto low=below_one.back(),high=above_one.back();below_one.pop_back();above_one.pop_back();
        table[low].threshold=static_cast<float>(std::clamp(scaled[low],0.,1.));table[low].alias=high;
        scaled[high]+=scaled[low]-1;
        (scaled[high]<1?below_one:above_one).push_back(high);
    }
    // Residual columns have unit probability (up to double rounding).
    for(auto i:below_one)table[i].threshold=1;
    for(auto i:above_one)table[i].threshold=1;
    return table;
}
} // namespace renderer
