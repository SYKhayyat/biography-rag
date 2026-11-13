# Dev shell for NixOS: PyPI wheels (numpy, onnxruntime, psycopg) need libstdc++ & zlib.
#   nix-shell --run 'uv run pytest'
{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  packages = [ pkgs.python313 ]; # uv: use your own (>= 0.13)
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
  UV_PYTHON_DOWNLOADS = "never";
}
