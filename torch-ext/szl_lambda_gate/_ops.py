# SPDX-License-Identifier: Apache-2.0
# Auto-style ops namespace shim for the universal kernel. Unique suffix lets
# multiple versions load in the same process (Kernel Hub requirement).
import torch

ops = torch.ops._szl_lambda_gate_20260623081355


def add_op_namespace_prefix(op_name: str) -> str:
    return f"_szl_lambda_gate_20260623081355::{op_name}"
