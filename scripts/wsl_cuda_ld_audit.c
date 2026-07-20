#define _GNU_SOURCE

#include <link.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// WSL exposes the Windows NVIDIA driver through /usr/lib/wsl/lib.  A Linux
// NVIDIA driver package installed in the distribution can leave another
// libcuda in /lib; injected CUDA developer tools may resolve that copy before
// the CUDA runtime does.  This glibc audit module changes only libcuda object
// searches and is never linked into the inference engine itself.
unsigned int la_version(unsigned int version) {
  (void)version;
  return LAV_CURRENT;
}

char* la_objsearch(const char* name, uintptr_t* cookie, unsigned int flag) {
  (void)cookie;
  (void)flag;
  if (name == NULL) {
    return NULL;
  }

  const char* basename = strrchr(name, '/');
  basename = basename == NULL ? name : basename + 1;
  // Do not redirect libcuda.so.1.1: the WSL loader opens that driver-specific
  // implementation after libcuda.so.1 has been selected.
  if (strcmp(basename, "libcuda.so") == 0 || strcmp(basename, "libcuda.so.1") == 0) {
    return "/usr/lib/wsl/lib/libcuda.so.1";
  }
  return (char*)name;
}
