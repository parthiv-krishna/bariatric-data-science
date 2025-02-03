{ pkgs ? import (fetchTarball https://github.com/nixos/nixpkgs/archive/refs/heads/nixos-24.11.tar.gz) {} }:
let
  python-packages = ps: with ps; [
    black
    numpy
    pandas
    polars
    pyarrow
    python-lsp-server
    scikit-learn
    scipy
    tqdm
  ];
  python-with-packages = pkgs.python3.withPackages python-packages;
in pkgs.mkShell {
  packages = [
    python-with-packages
  ];
}
