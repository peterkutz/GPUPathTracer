// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cutl/blob/master/LICENSE

#ifndef CUKD_PRIMITIVES_H
#define CUKD_PRIMITIVES_H

#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include "utils.h"

/**********************************************************************************
 *
 * Primitives, structs of device pointers
 *
 **********************************************************************************/

struct DevAABBArray {
    int length;
    UFloat4* minima, *maxima;
};

struct DevTriangleArray {
    int length;
    UFloat4 *v[3];
    UFloat4 *n[3];
    DevAABBArray aabbs;
};

struct DevRayArray {
    int length;
    UFloat4 *origins;
    UFloat4 *directions;
};

/**********************************************************************************
 *
 * AABBArray
 *
 **********************************************************************************/

struct AABBArray {
    typedef thrust::tuple<UFloat4, UFloat4> AABBTuple;
    typedef thrust::tuple<thrust::device_vector<UFloat4>::iterator,
                          thrust::device_vector<UFloat4>::iterator> AABBIteratorTuple;
    typedef thrust::zip_iterator<AABBIteratorTuple> AABBIterator;

    void resize(int size);

    int size() const;
    void clear();
    void copy(AABBArray & aabbs);

    AABBIterator begin();
    AABBIterator end();

    struct MinMax {
        __device__
        AABBTuple operator()(const AABBTuple & aabb1, const AABBTuple & aabb2) {
            return
                thrust::make_tuple(
                    f4min(thrust::get<0>(aabb1), thrust::get<0>(aabb2)),
                    f4max(thrust::get<1>(aabb1), thrust::get<1>(aabb2))
                );
        }
        Float4Maximum f4max;
        Float4Minimum f4min;
    };

    // calls cudaMemcpy, use sparingly
    void set(int index, const UFloat4 & min, const UFloat4 & max);
    void print(std::string prefix) const;

    DevAABBArray dev_array();

    DevVector<UFloat4> minima, maxima;
};

/**********************************************************************************
 *
 * Primitive host structs
 *
 **********************************************************************************/

struct Triangle {
    UFloat4 v[3];
    UFloat4 n[3];
};

struct AABB {
    float4 minimum, maximum;
};

struct cukdRay {
    UFloat4 origin;
    UFloat4 direction;
};

class RayArray {
    public:
        RayArray(const std::vector<cukdRay> & rays);
        void resize(int n) {
            origins.resize(n);
            directions.resize(n);
        };

        DevRayArray dev_array() {
            DevRayArray array;
            array.origins = origins.pointer();
            array.directions = directions.pointer();
            array.length = size();
            return array;
        };

        int size() const { return origins.size(); };
    private:
        DevVector<UFloat4> origins;
        DevVector<UFloat4> directions;
};

/**********************************************************************************
 *
 * TriangleArray
 *
 **********************************************************************************/

class TriangleArray {
    public:
        typedef std::shared_ptr<TriangleArray> Ptr;
        TriangleArray() {};
        TriangleArray(const std::vector<Triangle> & triangles);

        void compute_aabbs();

        void resize(int size);

        void append(const TriangleArray & tris);
        DevTriangleArray dev_array();

        void set(int index,
                 const float4 & vv1, const float4 & vv2, const float4 & vv3,
                 const float4 & nn1, const float4 & nn2, const float4 & nn3);

        int size() const;

        DevVector<UFloat4> v[3];
        DevVector<UFloat4> n[3];

        AABBArray aabbs;
};

/**********************************************************************************
 *
 * Kernel invocation wrappers
 *
 **********************************************************************************/

void triangle_aabbs(DevTriangleArray tris, int length);

/**********************************************************************************
 *
 * Inlines
 *
 **********************************************************************************/

inline int
AABBArray::size() const {
    return minima.size();
}

inline void
AABBArray::clear() {
    minima.clear();
    maxima.clear();
}

inline void
AABBArray::copy(AABBArray & aabbs) {
    aabbs.minima = minima.copy();
    aabbs.maxima = maxima.copy();
}

inline AABBArray::AABBIterator
AABBArray::begin() {
    return thrust::make_zip_iterator(thrust::make_tuple(minima.begin(), maxima.begin()));
}

inline AABBArray::AABBIterator
AABBArray::end() {
    return thrust::make_zip_iterator(thrust::make_tuple(minima.end(), maxima.end()));
}

inline int
TriangleArray::size() const {
    return v[0].size();
}

#endif  // CUKD_PRIMITIVES_H
