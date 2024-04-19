#include "runway_coords.h"

int main()
{
    auto threshs = get_thresholds_coords();
    for(auto thresh:threshs)
    {
        std::cout<<thresh<<std::endl;
    }
    return 0;
}