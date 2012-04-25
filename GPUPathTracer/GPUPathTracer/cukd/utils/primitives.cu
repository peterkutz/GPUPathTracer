// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cutl/blob/master/LICENSE

#include "../primitives.h"

/**********************************************************************************
 *
 * AABBArray implementation
 *
 **********************************************************************************/

DevAABBArray
AABBArray::dev_array() {
    DevAABBArray dev_aabb_array;
    dev_aabb_array.minima = minima.pointer();
    dev_aabb_array.maxima = maxima.pointer();
    dev_aabb_array.length = size();
    return dev_aabb_array;
}

void
AABBArray::resize(int size) {
    minima.resize(size);
    maxima.resize(size);
}

void
AABBArray::set(int index, const UFloat4 & min, const UFloat4 & max) {
    minima.set(index, min);
    maxima.set(index, max);
}

void
AABBArray::print(std::string prefix="") const {
    minima.print(prefix + "min");
    maxima.print(prefix + "max");
}

RayArray::RayArray(const std::vector<cukdRay> & rays) {
    std::vector<UFloat4> orig, dir;
    for(std::vector<cukdRay>::const_iterator it = rays.begin(); it != rays.end(); ++it) {
        orig.push_back(it->origin);
        dir.push_back(it->direction);
    }
    origins.populate(orig);
    directions.populate(dir);
};

/**********************************************************************************
 *
 * TriangleArray implementation
 *
 **********************************************************************************/

TriangleArray::TriangleArray(const std::vector<Triangle> & triangles) {
    std::vector<UFloat4> vv[3], nn[3];
    for(std::vector<Triangle>::const_iterator it = triangles.begin();
            it != triangles.end(); ++it) {
        for(int j = 0; j < 3; ++j) {
            vv[j].push_back(it->v[j]);
            nn[j].push_back(it->n[j]);
        }
    }
    for(int i = 0; i < 3; ++i) {
        v[i].populate(vv[i]);
        n[i].populate(nn[i]);
    }
    aabbs.resize(triangles.size());
}

void
TriangleArray::compute_aabbs() {
    DevTriangleArray dev_tris = dev_array();
    triangle_aabbs(dev_tris, size());
}

void 
TriangleArray::append(const TriangleArray & tris) {
    int old_size = size();
    resize(old_size + tris.size());
    for(int i = 0; i < 3; ++i) {
        thrust::copy(tris.v[i].begin(), tris.v[i].end(), v[i].begin() + old_size);
        thrust::copy(tris.n[i].begin(), tris.n[i].end(), n[i].begin() + old_size);
    }
}

DevTriangleArray
TriangleArray::dev_array() {
    DevTriangleArray dev_triangle_array;
    for(int i = 0; i < 3; ++i) {
        dev_triangle_array.v[i] = v[i].pointer();
        dev_triangle_array.n[i] = n[i].pointer();
    }
    dev_triangle_array.length = size();
    dev_triangle_array.aabbs = aabbs.dev_array();
    return dev_triangle_array;
}

void
TriangleArray::resize(int size) {
    for(int i = 0; i < 3; ++i) {
        v[i].resize(size);
        n[i].resize(size);
    }
    aabbs.resize(size);
}

// Trivial kernel for triangle AABB computation
__global__
void
triangle_aabbs_kernel(DevTriangleArray tris) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    UFloat4 min, max;
    if(tid < tris.length) {
        for(int i = 0; i < 3; ++i) {
            min.component[i] =
                min_three(tris.v[0][tid].component[i],
                          tris.v[1][tid].component[i],
                          tris.v[2][tid].component[i]);
            max.component[i] =
                max_three(tris.v[0][tid].component[i],
                          tris.v[1][tid].component[i],
                          tris.v[2][tid].component[i]);
        }

        tris.aabbs.minima[tid] = min;
        tris.aabbs.maxima[tid] = max;
    }
}

void
triangle_aabbs(DevTriangleArray tris, int length) {
    dim3 grid(IntegerDivide(256)(length),1,1);
    dim3 blocks(256, 1, 1);
    triangle_aabbs_kernel<<<grid,blocks>>>(tris);
}
