#ifndef EXECUTABLEPATH_HPP
#define EXECUTABLEPATH_HPP

#include <iostream>
#include <string>
#include <filesystem>

// Platform-specific includes
#ifdef _WIN32
    #include <windows.h>
#elif __linux__
    #include <unistd.h>
    #include <limits.h>
#elif __APPLE__
    #include <mach-o/dyld.h>
    #include <limits.h>
#endif

// Function to get executable path
std::string getExecutablePath() {
    char path[1024];

#ifdef _WIN32
    // Windows implementation
    GetModuleFileName(NULL, path, sizeof(path));
#elif __linux__
    // Linux implementation using /proc/self/exe
    ssize_t count = readlink("/proc/self/exe", path, sizeof(path));
    if (count != -1) {
        path[count] = '\0';  // Null-terminate the string
    }
#elif __APPLE__
    // macOS implementation using _NSGetExecutablePath
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) {
        return "";  // Buffer is too small
    }
#endif

    return std::string(path);
}

#endif // EXECUTABLEPATH_HPP