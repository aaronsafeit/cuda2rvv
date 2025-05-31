#ifndef CUDA2RVV_TEXTURE_H
#define CUDA2RVV_TEXTURE_H

#include <cstddef>
#include <cstdint>

// ----------------------
// Texture reference emulation
// ----------------------

// Texture fetch wrapper (simulate texture memory read)
template<typename T>
struct texture2D {
    const T* data;
    size_t width;
    size_t height;

    __host__ __device__
    texture2D() : data(nullptr), width(0), height(0) {}

    __host__ __device__
    texture2D(const T* ptr, size_t w, size_t h) : data(ptr), width(w), height(h) {}

    // Fetch with boundary check (clamp)
    __host__ __device__
    T fetch(int x, int y) const {
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (x >= (int)width) x = (int)width - 1;
        if (y >= (int)height) y = (int)height - 1;
        return data[y * width + x];
    }
};

// ----------------------
// Surface memory emulation
// ----------------------

// Surface write wrapper (simulate surface memory write)
template<typename T>
struct surface2D {
    T* data;
    size_t width;
    size_t height;

    __host__ __device__
    surface2D() : data(nullptr), width(0), height(0) {}

    __host__ __device__
    surface2D(T* ptr, size_t w, size_t h) : data(ptr), width(w), height(h) {}

    // Write with boundary check (clamp)
    __host__ __device__
    void write(int x, int y, T value) {
        if (x < 0 || y < 0 || x >= (int)width || y >= (int)height) return;
        data[y * width + x] = value;
    }
};

#endif // CUDA2RVV_TEXTURE_H
