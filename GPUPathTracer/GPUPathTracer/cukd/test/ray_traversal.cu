#include <ostream>
#include <vector_types.h>
#include <cutil_math.h>
#include <set>
#include <SDL/SDL.h>
#include <SDL/SDL_image.h>
#include "utils.h"
#include "kdtree.h"
#include "primitives.h"

TriangleArray load_ply(const std::string & filename) {
    std::ifstream file;
    file.open(filename.c_str());
    std::string line;
    bool data = false;

    int n_vertices = 0;
    int n_faces = 0;

    std::vector<float3> vertices, normals;
    std::vector<std::vector<int> > faces;
    std::string temp;
    while(file.good()) {
        getline(file, line);
        std::istringstream sstr(line);
        if(!data) {
            if(line.size() > 13 && line.compare(0,14,"element vertex") == 0) {
                sstr >> temp >> temp >> n_vertices;
            }
            if(line.size() > 11 && line.compare(0,12,"element face") == 0) {
                sstr >> temp >> temp >> n_faces;
            }
            if(line.size() > 9 && line.compare(0,10,"end_header") == 0) {
                data = true;
            }
        } else { // parse data
            if(n_vertices > 0) {
                float3 vertex, normal;
                sstr >> vertex.x >> vertex.y >> vertex.z
                     >> normal.x >> normal.y >> normal.z;
                vertices.push_back(vertex);
                normals.push_back(normal);
                n_vertices--;
            } else {
                int temp;
                int face[3];
                sstr >> temp >> face[0] >> face[1] >> face[2];
                std::vector<int> ff;
                for(int i = 0; i < 3; ++i)
                    ff.push_back(face[i]);
                faces.push_back(ff);

            }
        }
    }

    std::vector<Triangle> tris;
    for(int f = 0; f < faces.size(); ++f) {
        Triangle tri;
        for(int i = 0; i < 3; ++i) {
            tri.v[i].f3.vec = vertices[faces[f][i]];
            tri.n[i].f3.vec = normals[faces[f][i]];
        }
        tris.push_back(tri);
    }

    return TriangleArray(tris);
}
unsigned int pixel_from_color(const float3 & color) {
    return (unsigned int) (  ((int) round(color.x) << 16)
                           + ((int) round(color.y) << 8)
                           + round(color.z));
}

void put_pixel(const float3 & color, int x, int y, int width, unsigned int* buffer) {
    buffer[x + y*width] = pixel_from_color(color);
}

void write_to_bmp(int width, int height, std::vector<int> & hits, 
                  std::vector<int> cost) {
   std::string filename = "test.bmp";

   unsigned int* data_buffer = new unsigned int[width * height] ;
   for(int i = 0; i < width; i++) {
        for(int j = 0; j < height; ++j) {
            int index = width*j + i;
            if(hits[width*j + i] != -1)
                put_pixel(make_float3(cost[index],0,1.f), i, j, width, data_buffer);
            else
                put_pixel(make_float3(0,cost[index],0), i, j, width, data_buffer);
        }
    }

    SDL_Surface * image = SDL_CreateRGBSurfaceFrom(data_buffer, width, height,
            32, width*4, 0x00ff0000, 0x0000ff00, 0x000000ff, 0);

   SDL_SaveBMP(image, filename.c_str());
   SDL_FreeSurface(image);
}


void get_rays(int width, int height, float xmin, float xmax,
              float ymin, float ymax, std::vector<Ray> & rays_h) {
    Ray ray;
    for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            float x =  (xmax - xmin)/((float) width)*j - xmax;
            float y = - (ymax - ymin)/((float) height)*i + ymax;
            ray.direction = finite_ufloat4(make_ufloat4(0.f,0.f,1,0));
            ray.origin = make_ufloat4(x,y,-5,0);
            rays_h.push_back(ray);
        }
    }
}

int main(int argc, char* argv[]) {
    using namespace cukd;

    // create the tree
    std::cout << "Loading geometry ..." << std::endl;
    TriangleArray triarr = load_ply("test/dragon_res1.ply");
    triarr.compute_aabbs();

    std::cout << "Creating k-d tree for " << triarr.size() << " triangles"
              << std::endl;

    UAABB root_aabb;
    root_aabb.minimum = make_ufloat4(-3,-3,-10,0);
    root_aabb.maximum = make_ufloat4(3,3,1,0);

    std::cout << "    initializing tree.." << std::endl;
    KDTree kdtree(root_aabb, triarr, 64);

    std::cout << "    creating tree..." << std::endl;
    Timer full("total tree creation time: ");

    kdtree.create();
    kdtree.preorder();
    
    for(int i = 0; i < 10; ++i) {
        full.start();
        kdtree.clear(triarr);
        kdtree.create();
        kdtree.preorder();
        full.stop();
    }
    full.print();

    std::cout << "tree creation time average: " << full.get_ms()/10 << " ms" << std::endl;

    // run raytraversal and measure the time
    const int repetitions = 100;
    const int width = 1024, height = 864;
    int n_rays = width * height;

    std::vector<Ray> rays_h;
    float xmin = -0.1, xmax = 0.11;
    float ymin = 0.03, ymax = 0.23;
    get_rays(width, height, xmin, xmax, ymin, ymax, rays_h);

    RayArray rays(rays_h);

    DevVector<int> hits;
    DevVector<int> costs;
    DevVector<float> alpha, x1, x2;
    hits.resize(n_rays);
    costs.resize(n_rays);
    alpha.resize(n_rays);
    x1.resize(n_rays);
    x2.resize(n_rays);

    Timer traversal_time("total traversal time: ");
    std::cout << "traversing the tree " << repetitions << " times" << std::endl;
    traversal_time.start();
    for(int i = 0; i < repetitions; ++i)
        kdtree.ray_bunch_traverse(width, height, rays, hits, costs, alpha, x1, x2);
    traversal_time.stop();
    traversal_time.print();
    std::cout << "Total number of rays: " << n_rays << std::endl;
    std::cout << "Rays/sec:   " << (repetitions * 1000.f * n_rays/traversal_time.get_ms()) << std::endl;
    std::cout << "frames/sec: " << (repetitions * 1000.f / traversal_time.get_ms()) << std::endl;

    // write a bmp with the traversal cost
    std::vector<int> result, costh;
    thrust::copy(hits.begin(), hits.end(), std::back_inserter(result));
    thrust::copy(costs.begin(), costs.end(), std::back_inserter(costh));

    write_to_bmp(width, height, result, costh);
}
