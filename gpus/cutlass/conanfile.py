from conans import ConanFile, tools, CMake

class CutlassConan(ConanFile):
    name = "cutlass"
    version = "2.11.0"
    settings = "os", "compiler", "build_type", "arch"
    generators="cmake"

    def build(self):
        try:
            git = tools.Git(folder="./cutlass")
            git.clone("https://github.com/NVIDIA/cutlass.git","v2.11.0")
        except:
            pass
        cmake = CMake(self)
        cmake.definitions["CUTLASS_ENABLE_HEADERS_ONLY"] = "ON"
        cmake.definitions["CUTLASS_ENABLE_TESTS"] = "OFF"
        cmake.configure(source_folder=self.build_folder+"/cutlass")
        cmake.build()

    def package(self):
        try:
            git = tools.Git(folder="./cutlass")
            git.clone("https://github.com/NVIDIA/cutlass.git","v2.11.0")
        except:
            pass
        cmake = CMake(self)
        cmake.definitions["CUTLASS_ENABLE_HEADERS_ONLY"] = "ON"
        cmake.definitions["CUTLASS_ENABLE_TESTS"] = "OFF"
        cmake.configure(source_folder=self.build_folder+"/cutlass")

        cmake.install()

    def package_info(self):  # still very useful for package consumers
        self.cpp_info.names["cmake_find_package"] = "cutlass"

