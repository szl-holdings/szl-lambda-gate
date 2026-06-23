{
  description = "Flake for the SZL Holdings szl_lambda_gate universal kernel";
  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };
  outputs = { self, kernel-builder }:
    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
