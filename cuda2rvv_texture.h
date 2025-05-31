#ifndef CUDA2RVV_TEXTURE_H
#define CUDA2RVV_TEXTURE_H

#include <cstddef>
#include <cstdint>
#include <type_traits>

// ----------------------
// Texture Reference Emulation for LLVM Lowering
// ----------------------

/**
 * texture2D<T>
 * Emulates CUDA texture object for 2D data. Designed to integrate with
 * LLVM IR lowering infrastructure and metadata annotations.
 */
template<typename T>
struct texture2D {
    static_assert(std::is_trivially_copyable<T>::value, "texture2D<T> requires POD types");

    const T* data;          // Underlying pointer to texture data
    size_t width;
    size_t height;

    // Optional: metadata for LLVM lowering
    uint32_t tex_format;    // Placeholder for texture format info
    uint32_t tex_flags;     // Wrapping, filtering, etc.

    __host__ __device__
    texture2D() : data(nullptr), width(0), height(0), tex_format(0), tex_flags(0) {}

    __host__ __device__
    texture2D(const T* ptr, size_t w, size_t h, uint32_t fmt = 0, uint32_t flags = 0)
        : data(ptr), width(w), height(h), tex_format(fmt), tex_flags(flags) {}

    // Simulated fetch with clamp-to-edge boundary behavior
    __host__ __device__
    T fetch(int x, int y) const {
        if (!data) return T();

        x = (x < 0) ? 0 : (x >= static_cast<int>(width) ? static_cast<int>(width) - 1 : x);
        y = (y < 0) ? 0 : (y >= static_cast<int>(height) ? static_cast<int>(height) - 1 : y);
        return data[y * width + x];
    }
};

// ----------------------
// Surface Memory Emulation
// ----------------------

/**
 * surface2D<T>
 * Simulates CUDA 2D surface writes with boundary checking.
 * LLVM lowering may rewrite surface writes to vector scatter/store.
 */
template<typename T>
struct surface2D {
    static_assert(std::is_trivially_copyable<T>::value, "surface2D<T> requires POD types");

    T* data;
    size_t width;
    size_t height;

    // Optional surface metadata for future extensions
    uint32_t surface_flags;

    __host__ __device__
    surface2D() : data(nullptr), width(0), height(0), surface_flags(0) {}

    __host__ __device__
    surface2D(T* ptr, size_t w, size_t h, uint32_t flags = 0)
        : data(ptr), width(w), height(h), surface_flags(flags) {}

    // Simulated write with clamp-to-boundary
    __host__ __device__
    void write(int x, int y, T value) {
        if (!data) return;
        if (x < 0 || y < 0 || x >= static_cast<int>(width) || y >= static_cast<int>(height))
            return;
        data[y * width + x] = value;
    }
};

// ----------------------
// Future LLVM Integration Hooks
// ----------------------
// These can be extended with metadata annotations or IR lowering shims
// for texture intrinsics, vectorized fetch/store, and sampling extensions.

/*
 * Example placeholders (used in frontend or IR pass)
 *   - __llvm_texture_load(ptr, x, y)
 *   - __llvm_surface_store(ptr, x, y, val)
 */

#endif // CUDA2RVV_TEXTURE_H
