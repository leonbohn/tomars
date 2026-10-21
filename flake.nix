{
  description = "tomas: automatenbasiertes constrained decoding";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # rust-toolchain.toml pins the compiler so local builds and CI agree.
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
      ...
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems =
        f:
        nixpkgs.lib.genAttrs systems (
          system:
          f (
            import nixpkgs {
              inherit system;
              overlays = [ (import rust-overlay) ];
              # RX 5700 XT has no ROCm support; everything model-related runs
              # on CPU, so we don't need cudaSupport/rocmSupport here either.
              config.allowUnfree = false;
            }
          )
        );
    in
    {
      devShells = forAllSystems (pkgs: {
        default =
          let
            gpt2Tokenizer = pkgs.fetchurl {
              url = "https://huggingface.co/openai-community/gpt2/resolve/607a30d/tokenizer.json";
              hash = "sha256-hBTKuSTYubMwE/DSIcWGLzZe6b45xcK/ropanpcEeKY=";
            };
            toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
            python = pkgs.python3.withPackages (ps: [
              ps.torch
              ps.safetensors
            ]);
          in
          pkgs.mkShell {
            env.GPT2_TOKENIZER = "${gpt2Tokenizer}";
            packages = [
              toolchain
              pkgs.rust-analyzer
              pkgs.cargo-nextest
              pkgs.maturin
              python
            ];
          };
      });

      formatter = forAllSystems (pkgs: pkgs.nixfmt-tree);
    };
}
