### Alternate instructions for Windows, using vcpkg

You can acquire aubio through [vcpkg](https://github.com/Microsoft/vcpkg) from Microsoft, that version features all external libraries (libsndfile, libav etc) and support for the Windows store platform.

In the following tutorials, there are assumptions on target directories and platform names; change them accordingly to your needs.

#### Tutorial : install aubio

1. fork and clone vcpkg to `C:\vcpkg`
2. setup vcpkg: `bootstrap-vcpkg.bat`
3. integrate vcpkg to your system: `vcpkg integrate install` (note: requires admin on first use)
	- optional but recommended for an easy usage from within Visual Studio
4. get and build aubio: `vcpkg install aubio[:[x86|x64]-[windows|uwp]]`
    - the optional triplet suffix determines which platform you want
    - e.g. `x86-windows`, `x64-windows`, `x86-uwp`, `x64-uwp`
    - when no triplet is specified, the platform will be `x86-windows`
    - you can also install many triplets at once: `vcpkg install aubio:x86-uwp aubio:x64-uwp`

#### Tutorial : use aubio

Open Visual Studio and create a new C++ console application:

    #include "stdafx.h"
    #include "aubio/aubio.h"
    int main() {
      const auto vec = new_fvec(1234);
      // ...
      return 0;
    }
    
That's all, you can now run/debug your application, the vcpkg user-wide integration took care of setting includes/libraries directories, linking and deploying dependencies next to your executable.

#### Tutorial : generate a Visual Studio solution for aubio

This tutorial will cover the simple scenario of working against the code base of aubio fetched by vcpkg (from GitHub, Tag 0.4.6) and generating a solution for 64-bit desktop.

1. run `cmake-gui`
2. *Where is the source code:* `C:\vcpkg\buildtrees\aubio\src\aubio-0.4.6`
3. *Where to build the binaries:* `C:\vcpkg\buildtrees\aubio\src\aubio-0.4.6\vs2017-x64-windows`
4. press `Configure`
5. select `Visual Studio 15 2017 Win64` generator
6. select `Specify toolchain file for cross-compiling`
7. press `Next`
8. *Specify the Toolchain file:* `C:\vcpkg\scripts\buildsystems\vcpkg.cmake`
9. press `Finish`
10. point `CMAKE_INSTALL_PREFIX` to an appropriate directory: `C:\libs\aubio\x64-windows`
11. press `Generate`
12. open and build generated solution

You could also do that on a directory pointing to your own fork, change vcpkg portfile.cmake to point to your own fork, use the patching facility; there are many possibilities.

Find more about vcpkg at [https://github.com/Microsoft/vcpkg](https://github.com/Microsoft/vcpkg) and [https://vcpkg.readthedocs.io/](https://vcpkg.readthedocs.io/).
