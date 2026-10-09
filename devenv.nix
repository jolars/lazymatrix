{
  pkgs,
  ...
}:
{
  env.OPENBLAS_LP64_LIB = "${pkgs.openblasCompat}/lib";

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
  ];

  languages = {
    rust = {
      enable = true;
      toolchainFile = ./rust-toolchain.toml;
    };
  };

  git-hooks = {
    hooks = {
      clippy = {
        enable = true;

        settings = {
          allFeatures = true;
        };
      };

      rustfmt = {
        enable = true;
      };
    };
  };
}
