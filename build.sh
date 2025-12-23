#!/bin/bash

# EasiScriptX Build Script
# One-click build script for EasiScriptX

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
    print_status "Detected OS: $OS"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    case $OS in
        "linux")
            if command_exists apt-get; then
                sudo apt-get update
                sudo apt-get install -y build-essential cmake ninja-build
                sudo apt-get install -y libboost-all-dev libeigen3-dev
                sudo apt-get install -y libcurl4-openssl-dev
                sudo apt-get install -y libreadline-dev
                sudo apt-get install -y libtorch-dev
                sudo apt-get install -y libonnxruntime-dev
            elif command_exists yum; then
                sudo yum groupinstall -y "Development Tools"
                sudo yum install -y cmake ninja-build
                sudo yum install -y boost-devel eigen3-devel
                sudo yum install -y libcurl-devel
                sudo yum install -y readline-devel
            else
                print_warning "Package manager not found. Please install dependencies manually."
            fi
            ;;
        "macos")
            if command_exists brew; then
                brew install cmake ninja boost eigen curl readline
                brew install libtorch onnxruntime
            else
                print_warning "Homebrew not found. Please install dependencies manually."
            fi
            ;;
        "windows")
            print_warning "Windows dependencies should be installed via vcpkg or manually."
            ;;
        *)
            print_warning "Unknown OS. Please install dependencies manually."
            ;;
    esac
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    local missing_deps=()
    
    if ! command_exists cmake; then
        missing_deps+=("cmake")
    fi
    
    if ! command_exists make && ! command_exists ninja; then
        missing_deps+=("make or ninja")
    fi
    
    if ! command_exists g++ && ! command_exists clang++; then
        missing_deps+=("g++ or clang++")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_status "Installing missing dependencies..."
        install_dependencies
    else
        print_success "All dependencies found"
    fi
}

# Function to clean build directory
clean_build() {
    print_status "Cleaning build directory..."
    
    if [ -d "build" ]; then
        rm -rf build
        print_success "Build directory cleaned"
    fi
}

# Function to create build directory
create_build_dir() {
    print_status "Creating build directory..."
    mkdir -p build
    cd build
}

# Function to configure with CMake
configure_cmake() {
    print_status "Configuring with CMake..."
    
    local cmake_args=()
    
    # Add build type
    cmake_args+=("-DCMAKE_BUILD_TYPE=Release")
    
    # Add generator
    if command_exists ninja; then
        cmake_args+=("-GNinja")
        print_status "Using Ninja generator"
    else
        print_status "Using Make generator"
    fi
    
    # Add platform-specific options
    case $OS in
        "linux")
            cmake_args+=("-DCMAKE_CXX_COMPILER=g++")
            ;;
        "macos")
            cmake_args+=("-DCMAKE_CXX_COMPILER=clang++")
            ;;
        "windows")
            cmake_args+=("-DCMAKE_CXX_COMPILER=cl")
            ;;
    esac
    
    # Run CMake
    cmake "${cmake_args[@]}" ..
    
    if [ $? -eq 0 ]; then
        print_success "CMake configuration successful"
    else
        print_error "CMake configuration failed"
        exit 1
    fi
}

# Function to build the project
build_project() {
    print_status "Building project..."
    
    # Get number of CPU cores
    local cores
    if command_exists nproc; then
        cores=$(nproc)
    elif command_exists sysctl; then
        cores=$(sysctl -n hw.ncpu)
    else
        cores=4
    fi
    
    print_status "Using $cores parallel jobs"
    
    # Build
    if command_exists ninja; then
        ninja -j$cores
    else
        make -j$cores
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Build successful"
    else
        print_error "Build failed"
        exit 1
    fi
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    if [ -f "test_esx" ]; then
        ./test_esx
        if [ $? -eq 0 ]; then
            print_success "Tests passed"
        else
            print_error "Tests failed"
            exit 1
        fi
    else
        print_warning "Test executable not found"
    fi
}

# Function to run benchmarks
run_benchmarks() {
    print_status "Running benchmarks..."
    
    if [ -f "benchmark" ]; then
        ./benchmark
        if [ $? -eq 0 ]; then
            print_success "Benchmarks completed"
        else
            print_warning "Benchmarks failed"
        fi
    else
        print_warning "Benchmark executable not found"
    fi
}

# Function to run examples
run_examples() {
    print_status "Running examples..."
    
    if [ -f "esx" ]; then
        local examples=("train" "matmul" "distributed" "autonomic" "pretrained" "custom_losses")
        
        for example in "${examples[@]}"; do
            if [ -f "../examples/${example}.esx" ]; then
                print_status "Running example: $example"
                ./esx "../examples/${example}.esx"
            else
                print_warning "Example not found: $example"
            fi
        done
        
        print_success "Examples completed"
    else
        print_warning "ESX executable not found"
    fi
}

# Function to install the project
install_project() {
    print_status "Installing project..."
    
    if command_exists ninja; then
        ninja install
    else
        make install
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Installation successful"
    else
        print_warning "Installation failed"
    fi
}

# Function to show help
show_help() {
    echo "EasiScriptX Build Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -c, --clean         Clean build directory before building"
    echo "  -t, --test          Run tests after building"
    echo "  -b, --benchmark     Run benchmarks after building"
    echo "  -e, --examples      Run examples after building"
    echo "  -i, --install       Install the project after building"
    echo "  -a, --all           Run tests, benchmarks, examples, and install"
    echo "  -d, --deps          Install dependencies"
    echo "  --no-clean          Don't clean build directory"
    echo ""
    echo "Examples:"
    echo "  $0                  # Basic build"
    echo "  $0 -c -t            # Clean build and run tests"
    echo "  $0 -a               # Full build with all options"
    echo "  $0 -d               # Install dependencies only"
}

# Main function
main() {
    local clean=false
    local test=false
    local benchmark=false
    local examples=false
    local install=false
    local all=false
    local deps=false
    local no_clean=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--clean)
                clean=true
                shift
                ;;
            -t|--test)
                test=true
                shift
                ;;
            -b|--benchmark)
                benchmark=true
                shift
                ;;
            -e|--examples)
                examples=true
                shift
                ;;
            -i|--install)
                install=true
                shift
                ;;
            -a|--all)
                all=true
                shift
                ;;
            -d|--deps)
                deps=true
                shift
                ;;
            --no-clean)
                no_clean=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Set defaults based on --all flag
    if [ "$all" = true ]; then
        test=true
        benchmark=true
        examples=true
        install=true
    fi
    
    # Detect OS
    detect_os
    
    # Install dependencies if requested
    if [ "$deps" = true ]; then
        install_dependencies
        exit 0
    fi
    
    # Check dependencies
    check_dependencies
    
    # Clean build directory if requested
    if [ "$clean" = true ] && [ "$no_clean" = false ]; then
        clean_build
    fi
    
    # Create build directory
    create_build_dir
    
    # Configure with CMake
    configure_cmake
    
    # Build the project
    build_project
    
    # Run tests if requested
    if [ "$test" = true ]; then
        run_tests
    fi
    
    # Run benchmarks if requested
    if [ "$benchmark" = true ]; then
        run_benchmarks
    fi
    
    # Run examples if requested
    if [ "$examples" = true ]; then
        run_examples
    fi
    
    # Install if requested
    if [ "$install" = true ]; then
        install_project
    fi
    
    print_success "Build process completed successfully!"
    print_status "Executables are available in the build directory:"
    print_status "  - esx: Main EasiScriptX compiler/interpreter"
    print_status "  - test_esx: Test suite"
    print_status "  - benchmark: Performance benchmarks"
}

# Run main function with all arguments
main "$@"
