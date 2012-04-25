#ifndef CUTL_INTERSECTION_H
#define CUTL_INTERSECTION_H

#include <float.h>
#include "../utils.h"
#include "../primitives.h"

// defined as inlines, external functions not supported in kernels
__inline__ __host__ __device__
bool
intersect_aabb(const cukdRay & ray, const UAABB & aabb, float & p_near, float & p_far) {
    bool intersection = true;
    float p_near_result = -FLT_MAX;
    float p_far_result =  FLT_MAX;
    UFloat4 inv_dir;
    float p_near_comp, p_far_comp;

    inv_dir = inv_ufloat3(ray.direction);

    for(int i = 0; i < 3; ++i) {
        p_near_comp = (aabb.minimum.component[i] - ray.origin.component[i]) * inv_dir.component[i];
        p_far_comp = (aabb.maximum.component[i] - ray.origin.component[i]) * inv_dir.component[i];

        if(p_near_comp > p_far_comp) {
            float temp = p_near_comp;
            p_near_comp = p_far_comp;
            p_far_comp = temp;
        }

        p_near_result = ((p_near_comp > p_near_result) ? p_near_comp : p_near_result);
        p_far_result = ((p_far_comp < p_far_result) ? p_far_comp : p_far_result);

        if(p_near_result > p_far_result)
            intersection = false;
    }

    p_near = p_near_result;
    p_far = p_far_result;

    return intersection;
}

__inline__ __host__ __device__
bool
intersect_triangle(const cukdRay & ray, const Triangle & tri,
                   float & alpha, float & x1, float & x2) {

    float3 vec1, vec2, vec3;
    float3 e1 = diff_ufloat3(tri.v[1], tri.v[0]);
    float3 e2 = diff_ufloat3(tri.v[2], tri.v[0]);

    vec3 = cross(ray.direction.f3.vec, e2);

    float aabs = dot(e1, vec3);
    if(fabsf(aabs) < 1e-8f) {
        return false;
    }

    vec2 = diff_ufloat3(ray.origin, tri.v[0]);
    vec1 = cross(vec2, e1);

    x1 = dot(vec2,vec3)/aabs;
    x2 = dot(ray.direction.f3.vec, vec1)/aabs;
    alpha = dot(e2, vec1)/aabs;

    return ((x1 + x2) <= 1.f && x1 >= 0.f && x2 >= 0.f);
}

#endif  // CUTL_INTERSECTION_H
