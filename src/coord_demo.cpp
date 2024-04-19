#include "runway_coords.h"

int main()
{
    auto threshs = get_thresholds_coords();
    for(auto thresh:threshs)
    {
        for(auto point:thresh)
        {
            std::cout<<point<<std::endl;
        }
    }
    return 0;
}