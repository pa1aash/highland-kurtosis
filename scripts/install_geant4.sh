#!/bin/bash
set -euo pipefail

METHOD="${1:-conda}"
ENV_NAME="g4highland"

check_geant4() {
    echo "Checking Geant4 installation..."

    if conda info --envs 2>/dev/null | grep -q "${ENV_NAME}"; then
        echo "  Conda environment '${ENV_NAME}' exists"
        eval "$(conda shell.bash hook 2>/dev/null)" || true
        conda activate "${ENV_NAME}" 2>/dev/null || true
    fi

    if command -v geant4-config &>/dev/null; then
        echo "  Found: $(geant4-config --version)"
        echo "  Prefix: $(geant4-config --prefix)"

        local data_dir
        data_dir="$(geant4-config --prefix)/share/Geant4/data" 2>/dev/null || true

        if [ -d "${data_dir}" ]; then
            local n_datasets
            n_datasets=$(ls -d "${data_dir}"/*/ 2>/dev/null | wc -l | tr -d ' ')
            echo "  Data directory: ${data_dir}"
            echo "  Datasets installed: ${n_datasets}"
        fi

        if command -v root-config &>/dev/null; then
            echo "  ROOT: $(root-config --version)"
        else
            echo "  ROOT: not found (optional, needed for .root output)"
        fi

        echo "  Python packages:"
        for pkg in numpy scipy matplotlib uproot; do
            if python3 -c "import ${pkg}" 2>/dev/null; then
                echo "    ${pkg}: OK"
            else
                echo "    ${pkg}: MISSING"
            fi
        done

        return 0
    else
        echo "  Geant4 not found in PATH"
        echo ""
        echo "  To install: ./scripts/install_geant4.sh conda"
        return 1
    fi
}

print_env_setup() {
    echo "conda activate ${ENV_NAME}"
}

install_conda() {
    echo "Installing Geant4 via conda-forge"
    echo ""

    local CONDA_CMD=""
    if command -v mamba &>/dev/null; then
        CONDA_CMD="mamba"
        echo "Using mamba (faster solver)"
    elif command -v conda &>/dev/null; then
        CONDA_CMD="conda"
        echo "Using conda"
    else
        echo "ERROR: Neither conda nor mamba found."
        echo ""
        echo "Install Miniforge (recommended) or Miniconda:"
        echo "  macOS/Linux: https://github.com/conda-forge/miniforge#install"
        echo "  Or: brew install miniforge"
        echo ""
        echo "Then re-run this script."
        exit 1
    fi

    if conda info --envs 2>/dev/null | grep -q "${ENV_NAME}"; then
        echo ""
        echo "Environment '${ENV_NAME}' already exists."
        read -p "Recreate it? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing environment..."
            ${CONDA_CMD} env remove -n "${ENV_NAME}" -y
        else
            echo "Keeping existing environment. Run with --check to verify."
            return 0
        fi
    fi

    echo ""
    echo "Creating '${ENV_NAME}' environment with Geant4, ROOT, and Python tools..."
    echo "(This may take 5-15 minutes on first install)"
    echo ""

    ${CONDA_CMD} create -n "${ENV_NAME}" -c conda-forge -y \
        geant4 \
        root \
        cmake \
        compilers \
        python=3.12 \
        numpy \
        scipy \
        matplotlib \
        scikit-image \
        uproot \
        awkward

    echo ""
    echo "Installation complete!"
    echo ""
    echo "To use:"
    echo "  conda activate ${ENV_NAME}"
    echo ""
    echo "To build the simulation:"
    echo "  conda activate ${ENV_NAME}"
    echo "  mkdir -p build && cd build"
    echo "  cmake .."
    echo "  make -j\$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
    echo ""
    echo "To verify:"
    echo "  ./scripts/install_geant4.sh --check"
}

install_source() {
    echo "Building Geant4 from source"
    echo ""
    echo "NOTE: conda-forge installation is recommended instead."
    echo "      Run: ./scripts/install_geant4.sh conda"
    echo ""

    GEANT4_VERSION="11.2.2"
    INSTALL_DIR="${HOME}/geant4/install"
    BUILD_DIR="${HOME}/geant4/build"
    SRC_DIR="${HOME}/geant4/src"

    for cmd in cmake make curl; do
        if ! command -v ${cmd} &>/dev/null; then
            echo "ERROR: ${cmd} not found. Install it first."
            exit 1
        fi
    done

    mkdir -p "${SRC_DIR}" "${BUILD_DIR}" "${INSTALL_DIR}"

    echo "Downloading Geant4 ${GEANT4_VERSION}..."
    cd "${SRC_DIR}"
    if [ ! -d "geant4-v${GEANT4_VERSION}" ]; then
        curl -L "https://github.com/Geant4/geant4/archive/refs/tags/v${GEANT4_VERSION}.tar.gz" \
            -o "geant4-v${GEANT4_VERSION}.tar.gz"
        tar xzf "geant4-v${GEANT4_VERSION}.tar.gz"
    fi

    echo "Building (this will take 30-60 minutes)..."
    cd "${BUILD_DIR}"
    cmake "${SRC_DIR}/geant4-v${GEANT4_VERSION}" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DGEANT4_INSTALL_DATA=ON \
        -DGEANT4_USE_OPENGL_X11=OFF \
        -DGEANT4_USE_QT=OFF \
        -DGEANT4_BUILD_MULTITHREADED=ON \
        -DCMAKE_BUILD_TYPE=Release

    make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
    make install

    echo ""
    echo "Geant4 installed to: ${INSTALL_DIR}"
    echo ""
    echo "Add to your shell profile:"
    echo "  source ${INSTALL_DIR}/bin/geant4.sh"
    echo "  export PATH=${INSTALL_DIR}/bin:\$PATH"
}

install_python_deps() {
    echo ""
    echo "Installing Python dependencies"

    pip3 install numpy scipy matplotlib scikit-image numpy-stl uproot awkward 2>/dev/null || \
    pip install numpy scipy matplotlib scikit-image numpy-stl uproot awkward 2>/dev/null || {
        echo "WARNING: Could not install Python packages."
        echo "Install manually: pip install numpy scipy matplotlib scikit-image numpy-stl uproot awkward"
    }
}

case "${METHOD}" in
    --check|check)
        check_geant4
        ;;
    --env-setup|env-setup)
        print_env_setup
        ;;
    conda|--conda)
        install_conda
        echo ""
        echo "Verifying installation..."
        eval "$(conda shell.bash hook 2>/dev/null)" || true
        conda activate "${ENV_NAME}" 2>/dev/null || true
        check_geant4 || true
        ;;
    source|--source)
        install_source
        check_geant4 || true
        install_python_deps
        ;;
    *)
        echo "MCS Highland - Geant4 Installation Helper"
        echo ""
        echo "Usage: $0 [METHOD]"
        echo ""
        echo "Methods:"
        echo "  conda        Install via conda-forge (recommended, default)"
        echo "  source       Build Geant4 from source"
        echo ""
        echo "Other options:"
        echo "  --check      Check if Geant4 is installed correctly"
        echo "  --env-setup  Print shell environment setup commands"
        echo ""
        echo "Examples:"
        echo "  $0                  # Install via conda-forge"
        echo "  $0 conda           # Same as above"
        echo "  $0 --check         # Verify installation"
        exit 1
        ;;
esac
