{
  description = "NOSBench-101: Towards Reproducible Neural Optimizer Search";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudaVersion = "12";
          };
        };

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

        python = pkgs.python312;
        pythonPackages = pkgs.python312Packages;

        pyprojectOverrides = final: prev: {
          configspace = prev.configspace.overrideAttrs {
              nativeBuildInputs =
              prev.configspace.nativeBuildInputs
              ++ [
                (final.resolveBuildSystem {
                  setuptools = [ ];
                  # scikit-build = [ ];
                  # cmake = [ ];
                  # ninja = [ ];
                })
              ]
              ++ [
                # pkgs.spdlog
                # pkgs.eigen
                # pkgs.suitesparse
                # pythonPackages.pybind11
                # pkgs.patchelf
              ];
          };
          nvidia-cufile-cu12 = prev.nvidia-cufile-cu12.overrideAttrs {
              nativeBuildInputs =
              prev.nvidia-cufile-cu12.nativeBuildInputs
              ++ [
                  pkgs.cudaPackages_12.libcufile
          	      pkgs.cudaPackages_12.cudatoolkit
          	      pkgs.cudaPackages_12.cuda_cudart
          	      pkgs.cudaPackages_12.cudnn
              ];
          };
          nvidia-cusolver-cu12 = prev.nvidia-cusolver-cu12.overrideAttrs {
              nativeBuildInputs =
              prev.nvidia-cusolver-cu12.nativeBuildInputs
              ++ [
          	      pkgs.cudaPackages_12.libcublas
          	      pkgs.cudaPackages_12.libcusparse
          	      pkgs.cudaPackages_12.libnvjitlink
              ];
          };
          nvidia-cusparse-cu12 = prev.nvidia-cusparse-cu12.overrideAttrs {
              nativeBuildInputs =
              prev.nvidia-cusparse-cu12.nativeBuildInputs
              ++ [
          	      pkgs.cudaPackages_12.libcublas
          	      pkgs.cudaPackages_12.libnvjitlink
              ];
          };
          torch = prev.torch.overrideAttrs {
              nativeBuildInputs =
              prev.torch.nativeBuildInputs
              ++ [
                  pkgs.cudaPackages_12.cudnn
                  pkgs.cudaPackages_12.cuda_cccl
                  pkgs.cudaPackages_12.cuda_cudart
                  pkgs.cudaPackages_12.cuda_cupti
                  pkgs.cudaPackages_12.cuda_nvcc
                  pkgs.cudaPackages_12.cuda_nvml_dev
                  pkgs.cudaPackages_12.cuda_nvrtc
                  pkgs.cudaPackages_12.cuda_nvtx
                  pkgs.cudaPackages_12.libcublas
                  pkgs.cudaPackages_12.libcufft
                  pkgs.cudaPackages_12.libcufile
                  pkgs.cudaPackages_12.libcurand
                  pkgs.cudaPackages_12.libcusolver
                  pkgs.cudaPackages_12.libcusparse
                  pkgs.cudaPackages_12.nccl
                  pkgs.cudaPackages_12.cusparselt
              ];
          };
        };

        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
            stdenv = pkgs.stdenv.override {
              targetPlatform = pkgs.stdenv.targetPlatform // {
                darwinSdkVersion = "15.1";
              };
            };
          }).overrideScope
            (
              pkgs.lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                overlay
                pyprojectOverrides
              ]
            );
      in
      {
        devShells.default =
          let
            editableOverlay = workspace.mkEditablePyprojectOverlay {
              root = "$REPO_ROOT";
              members = [ "nosbench" ];
            };

            editablePythonSet = pythonSet.overrideScope (
              pkgs.lib.composeManyExtensions [
                editableOverlay

                (final: prev: {
                  nosbench = prev.nosbench.overrideAttrs (old: {
                    src = pkgs.lib.fileset.toSource {
                      root = old.src;
                      fileset = pkgs.lib.fileset.unions [
                        (old.src + "/pyproject.toml")
                        (old.src + "/README.md")
                      ];
                    };

                    nativeBuildInputs = old.nativeBuildInputs ++ final.resolveBuildSystem { editables = [ ]; };
                  });
                })
              ]
            );

            virtualenv = editablePythonSet.mkVirtualEnv "nosbench-dev-env" workspace.deps.all;
          in
          pkgs.mkShell {
            packages = [
              virtualenv
              pythonPackages.uv
              pythonPackages.ruff
              pythonPackages.python-lsp-server
            ];

            env = {
              UV_NO_SYNC = "1";
              UV_PYTHON = "${virtualenv}/bin/python";
              UV_PYTHON_DOWNLOADS = "never";
            };

            shellHook = ''
              unset PYTHONPATH
              export REPO_ROOT=$(git rev-parse --show-toplevel)
            '';
          };
      }
    );
}
