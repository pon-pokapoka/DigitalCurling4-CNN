// Copyright (c) 2023-2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT

#ifndef EXECUTABLEPATH_HPP
#define EXECUTABLEPATH_HPP

#include <iostream>
#include <string>
#include <filesystem>
#include <system_error>

// Platform-specific includes
#ifdef _WIN32
    #include <windows.h>
#elif __APPLE__
    #include <mach-o/dyld.h>
    #include <limits.h>
#endif

// Function to get executable path. Returns an empty string on failure.
std::string getExecutablePath() {
#ifdef _WIN32
    // Windows implementation
    char path[1024];
    DWORD const length = GetModuleFileName(NULL, path, sizeof(path));
    if (length == 0 || length >= sizeof(path)) return "";
    return std::string(path, length);
#elif __linux__
    // Linux implementation using /proc/self/exe
    std::error_code ec;
    auto const path = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::string() : path.string();
#elif __APPLE__
    // macOS implementation using _NSGetExecutablePath
    char path[1024];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) {
        return "";  // Buffer is too small
    }
    return std::string(path);
#else
    return "";
#endif
}

#endif // EXECUTABLEPATH_HPP