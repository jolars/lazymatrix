{
  config,
  lib,
  pkgs,
  ...
}:
let
  rustOverlay = config.lib.getInput {
    name = "rust-overlay";
    url = "github:oxalica/rust-overlay";
    attribute = "DIPLODOCUS_RUST_TOOLCHAIN";
    follows = [ "nixpkgs" ];
  };
  rustDocumentationToolchain =
    (rustOverlay.lib.mkRustBin { } pkgs.buildPackages).nightly."2026-09-28".minimal;
in
{
  env.OPENBLAS_LP64_LIB = "${pkgs.openblasCompat}/lib";
  env.DIPLODOCUS_RUST_TOOLCHAIN = rustDocumentationToolchain;
  env.UV_PROJECT_ENVIRONMENT = "${config.devenv.root}/.devenv/python";
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];

  # npm's workerd binary needs an explicit loader on NixOS.
  env.MINIFLARE_WORKERD_PATH = pkgs.writeShellScript "lazymatrix-workerd" ''
    runtime=$(node -p "require(require.resolve('workerd', {paths: [require.resolve('wrangler/package.json')]})).default")
    exec "${pkgs.stdenv.cc.bintools.dynamicLinker}" \
      --library-path "${pkgs.glibc}/lib" "$runtime" "$@"
  '';

  packages = with pkgs; [
    go-task
    llvmPackages.bintools
    liteparse
    cargo-llvm-cov
    cargo-flamegraph
    cargo-audit
    cargo-deny
    cargo-msrv
    gnuplot
    samply
    pprof
    wasm-pack
    perf
    go-task
    quartoMinimal
    shfmt
    nodejs_24
    pnpm_10
    maturin
    ruff
    uv
  ];

  languages = {
    rust = {
      enable = true;
      toolchainFile = ./rust-toolchain.toml;
    };
    python.enable = true;
  };

  git-hooks = {
    hooks = {
      clippy = {
        enable = true;

        settings = {
          allFeatures = true;
        };
      };

      prettier = {
        enable = true;
        files = "^docs/";

        settings = {
          binPath = "./docs/node_modules/.bin/prettier";
          ignore-path = [ "docs/.prettierignore" ];
        };
      };

      rustfmt = {
        enable = true;
      };
    };
  };
}
